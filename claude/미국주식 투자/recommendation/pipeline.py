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


def _bulk_download_chunk(
    tickers_chunk: list[str],
    period: str,
    max_retries: int = 3,
    backoff_base: float = 5.0,
) -> dict[str, pd.DataFrame]:
    """yf.download로 한 청크를 일괄 다운로드 후 티커별 DataFrame 딕셔너리 반환.

    rate limit 시 backoff_base * 2^attempt 초 대기 후 재시도.
    """
    import time as _time

    if not tickers_chunk:
        return {}

    raw = None
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers_chunk,
                period=period,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
                timeout=30,
            )
            if not raw.empty:
                break
        except Exception as e:
            last_exc = e

        wait = backoff_base * (2 ** attempt)
        print(f"  rate-limited or empty, retry {attempt + 1}/{max_retries} after {wait:.0f}s")
        _time.sleep(wait)

    if raw is None or raw.empty:
        if last_exc:
            print(f"WARN: bulk download failed for chunk: {last_exc}")
        return {}

    out: dict[str, pd.DataFrame] = {}
    if len(tickers_chunk) == 1:
        # 단일 티커일 때는 multi-index 아닌 일반 DataFrame
        if not raw.empty:
            out[tickers_chunk[0]] = raw.dropna(how="all")
        return out

    for ticker in tickers_chunk:
        if ticker not in raw.columns.get_level_values(0):
            continue
        df = raw[ticker].dropna(how="all")
        if not df.empty:
            out[ticker] = df
    return out


def batch_fetch(
    tickers: list[str],
    period: str = "1y",
    max_workers: int = 10,
    use_cache: bool = True,
    chunk_size: int = 100,
) -> list[dict]:
    """
    여러 종목 데이터를 yf.download 벌크 호출로 수집한다.

    Args:
        tickers: 티커 리스트
        period: 데이터 기간
        max_workers: (사용 안 함 — yf.download가 내부 멀티스레드 처리)
        use_cache: 당일 캐시 사용 여부
        chunk_size: yf.download 한 호출당 티커 수 (yfinance 권장 ~100)

    Returns:
        성공한 종목 데이터 리스트. info는 빈 dict (벌크 모드는 가격만).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    to_fetch: list[str] = []
    today = date.today().isoformat()

    if use_cache:
        for ticker in tickers:
            cache_file = CACHE_DIR / f"{ticker}_{today}.parquet"
            if cache_file.exists():
                try:
                    price_df = pd.read_parquet(cache_file)
                    if len(price_df) >= 50:
                        results.append({"ticker": ticker, "price_df": price_df, "info": {}})
                        continue
                except Exception:
                    pass
            to_fetch.append(ticker)
    else:
        to_fetch = list(tickers)

    if not to_fetch:
        return results

    print(f"데이터 수집: {len(to_fetch)}개 종목 (캐시 히트: {len(results)}개), 청크 크기 {chunk_size}")

    chunks = [to_fetch[i:i + chunk_size] for i in range(0, len(to_fetch), chunk_size)]
    for chunk in tqdm(chunks, desc="Bulk download"):
        data_by_ticker = _bulk_download_chunk(chunk, period)
        for ticker, df in data_by_ticker.items():
            if len(df) < 50:
                continue
            cache_file = CACHE_DIR / f"{ticker}_{today}.parquet"
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
            results.append({"ticker": ticker, "price_df": df, "info": {}})

    print(f"수집 완료: {len(results)}개 종목 성공 / {len(tickers)}개 전체")
    return results


def run_pipeline(
    period: str = "1y",
    max_workers: int = 10,
    top_n: int = 30,
    use_cache: bool = True,
    universe_subset: list[str] | None = None,
    exemplar_weight: float = 0.0,
    top_mcap: int | None = None,
    min_mcap: float = 0.0,
) -> pd.DataFrame:
    """
    전체 추천 파이프라인을 실행한다.

    Args:
        period: 가격 데이터 기간
        max_workers: 병렬 스레드 수
        top_n: 최종 추출할 상위 종목 수
        use_cache: 캐시 사용 여부
        universe_subset: 특정 종목 리스트 (None이면 전체 유니버스)
        exemplar_weight: 7번째 팩터 가중치
        top_mcap: 시가총액 상위 N개로 제한 (None이면 전체)
        min_mcap: 최소 시가총액 (USD)

    Returns:
        전체 종목 스코어 DataFrame (정렬됨)
    """
    # 1. 유니버스 로드
    if universe_subset:
        tickers = universe_subset
        print(f"커스텀 유니버스: {len(tickers)}개 종목")
    else:
        universe_df = get_universe(top_n_by_mcap=top_mcap, min_market_cap=min_mcap)
        tickers = universe_df["ticker"].tolist()
        if top_mcap:
            print(f"시총 상위 {top_mcap}개 유니버스: {len(tickers)}개 종목")
        else:
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
    parser.add_argument("--top-mcap", type=int, default=None,
                        help="시가총액 상위 N개로 유니버스 제한 (예: 3000)")
    parser.add_argument("--min-mcap", type=float, default=0.0,
                        help="최소 시가총액 (USD, 예: 1e8=1억달러)")

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
        top_mcap=args.top_mcap,
        min_mcap=args.min_mcap,
    )
