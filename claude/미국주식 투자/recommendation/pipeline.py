"""
추천 시스템 파이프라인 오케스트레이션

전체 흐름:
1. 유니버스 로드
2. 가격+펀더멘털 데이터 배치 수집
3. 6팩터 스코어링
4. Top N 추출
5. 결과 저장
"""

import pandas as pd
import yfinance as yf
import numpy as np
from pathlib import Path
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from recommendation.universe import get_universe
from recommendation.exemplar.library import ExemplarLibrary
from recommendation.exemplar.profile import load_profile
from recommendation.scorer import RecommendationScorer, build_weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SCORES_DIR = PROJECT_ROOT / "data" / "scores"
EXEMPLAR_DIR = PROJECT_ROOT / "data" / "exemplars"
EXEMPLAR_JSON = EXEMPLAR_DIR / "exemplars.json"


def load_active_profiles() -> list[tuple[str, dict]]:
    """라이브러리에서 활성 모범의 (id, profile) 리스트를 로드한다."""
    if not EXEMPLAR_JSON.exists():
        return []
    lib = ExemplarLibrary(EXEMPLAR_JSON)
    actives = lib.list_all(active_only=True)
    out: list[tuple[str, dict]] = []
    for ex in actives:
        profile_path = PROJECT_ROOT / ex.profile_path
        if not profile_path.exists():
            print(f"WARN: profile missing for {ex.id} at {profile_path}, skipping")
            continue
        out.append((ex.id, load_profile(profile_path)))
    return out


def fetch_ticker_data(ticker: str, period: str = "1y") -> dict:
    """
    단일 종목의 가격 + 펀더멘털 데이터를 수집한다.

    Returns:
        {"ticker": str, "price_df": DataFrame, "info": dict} or None on failure
    """
    try:
        t = yf.Ticker(ticker)
        price_df = t.history(period=period, auto_adjust=True)

        if price_df.empty or len(price_df) < 50:
            return None

        info = t.info if hasattr(t, "info") else {}

        return {
            "ticker": ticker,
            "price_df": price_df,
            "info": info if info else {},
        }
    except Exception:
        return None


def batch_fetch(
    tickers: list[str],
    period: str = "1y",
    max_workers: int = 10,
    use_cache: bool = True,
) -> list[dict]:
    """
    여러 종목 데이터를 병렬로 수집한다.

    Args:
        tickers: 티커 리스트
        period: 데이터 기간
        max_workers: 병렬 스레드 수
        use_cache: 캐시 사용 여부

    Returns:
        성공한 종목 데이터 리스트
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    to_fetch = []

    if use_cache:
        today = date.today().isoformat()
        for ticker in tickers:
            cache_file = CACHE_DIR / f"{ticker}_{today}.parquet"
            if cache_file.exists():
                try:
                    price_df = pd.read_parquet(cache_file)
                    results.append({
                        "ticker": ticker,
                        "price_df": price_df,
                        "info": {},  # 캐시에서는 info 없음
                    })
                    continue
                except Exception:
                    pass
            to_fetch.append(ticker)
    else:
        to_fetch = tickers

    if not to_fetch:
        return results

    print(f"데이터 수집: {len(to_fetch)}개 종목 (캐시 히트: {len(results)}개)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_ticker_data, ticker, period): ticker
            for ticker in to_fetch
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching"):
            result = future.result()
            if result:
                # 캐시 저장
                ticker = result["ticker"]
                today = date.today().isoformat()
                cache_file = CACHE_DIR / f"{ticker}_{today}.parquet"
                try:
                    result["price_df"].to_parquet(cache_file)
                except Exception:
                    pass
                results.append(result)

    print(f"수집 완료: {len(results)}개 종목 성공 / {len(tickers)}개 전체")
    return results


def run_pipeline(
    period: str = "1y",
    max_workers: int = 10,
    top_n: int = 30,
    use_cache: bool = True,
    universe_subset: list[str] | None = None,
    exemplar_weight: float = 0.0,
) -> pd.DataFrame:
    """
    전체 추천 파이프라인을 실행한다.

    Args:
        period: 가격 데이터 기간
        max_workers: 병렬 스레드 수
        top_n: 최종 추출할 상위 종목 수
        use_cache: 캐시 사용 여부
        universe_subset: 특정 종목 리스트 (None이면 전체 유니버스)

    Returns:
        전체 종목 스코어 DataFrame (정렬됨)
    """
    # 1. 유니버스 로드
    if universe_subset:
        tickers = universe_subset
        print(f"커스텀 유니버스: {len(tickers)}개 종목")
    else:
        universe_df = get_universe()
        tickers = universe_df["ticker"].tolist()
        print(f"전체 유니버스: {len(tickers)}개 종목")

    # 2. 데이터 수집
    data_list = batch_fetch(tickers, period=period, max_workers=max_workers, use_cache=use_cache)

    if not data_list:
        print("수집된 데이터 없음. 파이프라인 종료.")
        return pd.DataFrame()

    # 3. 스코어링
    profiles = load_active_profiles() if exemplar_weight > 0 else []
    if profiles:
        print(f"활성 모범: {len(profiles)}개 (exemplar_weight={exemplar_weight})")
        weights = build_weights(exemplar_weight=exemplar_weight)
    else:
        weights = build_weights(exemplar_weight=0.0)
    scorer = RecommendationScorer(weights=weights, profiles=profiles)
    scores_df = scorer.score_all(data_list)

    # 4. 정렬 및 저장
    scores_df = scores_df.sort_values("total_score", ascending=False).reset_index(drop=True)
    scores_df["rank"] = range(1, len(scores_df) + 1)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    output_file = SCORES_DIR / f"{today}_scores.parquet"
    scores_df.to_parquet(output_file, index=False)
    print(f"\n스코어 저장: {output_file}")
    print(f"Top {top_n} 종목:")
    display_cols = ["rank", "ticker", "total_score", "momentum", "trend", "breakout", "valuation", "growth", "risk"]
    if "exemplar_similarity" in scores_df.columns:
        display_cols.append("exemplar_similarity")
        display_cols.append("best_match_id")
    print(scores_df.head(top_n)[display_cols].to_string())

    return scores_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="종목 추천 파이프라인 실행")
    parser.add_argument("--period", default="1y", help="데이터 기간 (기본: 1y)")
    parser.add_argument("--workers", type=int, default=10, help="병렬 스레드 수")
    parser.add_argument("--top", type=int, default=30, help="상위 N개 표시")
    parser.add_argument("--no-cache", action="store_true", help="캐시 미사용")
    parser.add_argument("--subset", type=str, default=None, help="종목 리스트 (콤마 구분)")
    parser.add_argument("--exemplar-weight", type=float, default=0.0,
                        help="7번째 팩터(모범 유사도) 가중치 0~0.4 (기본 0 = 비활성)")

    args = parser.parse_args()

    subset = None
    if args.subset:
        subset = [t.strip().upper() for t in args.subset.split(",")]

    run_pipeline(
        period=args.period,
        max_workers=args.workers,
        top_n=args.top,
        use_cache=not args.no_cache,
        universe_subset=subset,
        exemplar_weight=args.exemplar_weight,
    )
