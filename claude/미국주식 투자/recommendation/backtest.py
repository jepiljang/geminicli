"""
추천 시스템 워크포워드 백테스트.

- 주간(또는 임의) 리밸런싱: 매 리밸런스 시점에 과거 데이터만으로 스코어링 → Top N equal-weight 매수.
- Look-ahead 방지: 각 ticker의 DataFrame을 df.loc[:rebalance_date]로 슬라이스해서 score_single 호출.
- 벤치마크: SPY equity curve 동기간 산출.
- 결과 메트릭: backtest.metrics 모듈 재사용 (CAGR, Sharpe, MDD, Win Rate, Alpha).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date as date_cls, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from recommendation.pipeline import CACHE_DIR, batch_fetch
from recommendation.scorer import FACTOR_WEIGHTS, RecommendationScorer
from recommendation.universe import get_universe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtest"


def _load_cached_prices(tickers: list[str], today_iso: str) -> dict[str, pd.DataFrame]:
    """파이프라인이 만든 당일 캐시(_today.parquet)에서 가격을 일괄 로드."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        path = CACHE_DIR / f"{t}_{today_iso}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                out[t] = df
            except Exception:
                continue
    return out


def _get_spy(period: str = "1y") -> pd.DataFrame:
    """SPY 일봉 가격 (벤치마크용)."""
    df = yf.download("SPY", period=period, auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise RuntimeError("SPY price fetch failed")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _score_ticker_sliced(args: tuple[str, pd.DataFrame, pd.Timestamp]) -> dict | None:
    """워커 함수: 한 ticker의 df를 cutoff까지 슬라이스 후 score_single."""
    ticker, df, cutoff = args
    sub = df.loc[:cutoff]
    if len(sub) < 50:
        return None
    scorer = RecommendationScorer()
    return scorer.score_single({"ticker": ticker, "price_df": sub, "info": {}})


def _rank_top_n_at(
    price_data: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
    top_n: int,
) -> list[str]:
    """cutoff 시점에서 전 종목 스코어링 → 백분위 정규화 → Top N 티커."""
    raw: list[dict] = []
    scorer = RecommendationScorer()
    for ticker, df in price_data.items():
        sub = df.loc[:cutoff]
        if len(sub) < 50:
            continue
        r = scorer.score_single({"ticker": ticker, "price_df": sub, "info": {}})
        if r:
            raw.append(r)
    if not raw:
        return []
    df_scores = pd.DataFrame(raw)
    factor_cols = ["momentum", "trend", "breakout", "valuation", "growth", "risk"]
    for col in factor_cols:
        if col in df_scores.columns:
            df_scores[col] = df_scores[col].rank(pct=True) * 100
    df_scores["total_score"] = sum(
        df_scores[c] * FACTOR_WEIGHTS[c] for c in factor_cols if c in df_scores.columns
    )
    df_scores = df_scores.sort_values("total_score", ascending=False)
    return df_scores.head(top_n)["ticker"].tolist()


def _close_on_or_before(df: pd.DataFrame, d: pd.Timestamp) -> float | None:
    """d에 종가 있으면 그 값, 없으면 d 직전의 마지막 종가. 둘 다 없으면 None."""
    sub = df.loc[:d]
    if sub.empty:
        return None
    val = sub["Close"].iloc[-1]
    return float(val) if not np.isnan(val) else None


def run_backtest(
    price_data: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    top_n: int = 10,
    rebalance_freq: str = "W-MON",
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """주간 리밸런싱 백테스트 실행.

    Returns:
        {"equity_curve", "spy_equity", "rebalances", "summary"}
    """
    rebalance_dates = pd.date_range(start_date, end_date, freq=rebalance_freq)

    # 전체 거래일 합집합 (모든 ticker + SPY)
    all_dates = set(spy_df.index)
    for df in price_data.values():
        all_dates.update(df.index)
    all_dates = sorted(d for d in all_dates if start_date <= d <= end_date)

    rebalance_set = {pd.Timestamp(d).normalize() for d in rebalance_dates}

    cash = initial_capital
    positions: dict[str, float] = {}  # {ticker: shares}
    equity_history: list[tuple[pd.Timestamp, float]] = []
    rebalances: list[dict[str, Any]] = []

    for d in tqdm(all_dates, desc="Backtest"):
        is_rebalance = pd.Timestamp(d).normalize() in rebalance_set

        if is_rebalance:
            # 현재 보유분 청산
            for t, sh in positions.items():
                px = _close_on_or_before(price_data[t], d)
                if px is not None:
                    cash += sh * px
            positions = {}

            # 새 Top N 픽
            picks = _rank_top_n_at(price_data, d, top_n)
            if picks:
                per_ticker = cash / len(picks)
                bought = []
                for t in picks:
                    px = _close_on_or_before(price_data[t], d)
                    if px is None or px <= 0:
                        continue
                    shares = per_ticker / px
                    positions[t] = shares
                    cash -= shares * px
                    bought.append(t)
                rebalances.append({"date": d, "tickers": bought, "equity": cash + sum(
                    sh * _close_on_or_before(price_data[t], d) or 0 for t, sh in positions.items()
                )})

        # Mark-to-market
        equity = cash
        for t, sh in positions.items():
            px = _close_on_or_before(price_data[t], d)
            if px is not None:
                equity += sh * px
        equity_history.append((d, equity))

    equity_curve = pd.Series(
        [e for _, e in equity_history],
        index=[d for d, _ in equity_history],
        name="equity",
    )

    # SPY benchmark equity curve (initial_capital 기준 정규화)
    spy_window = spy_df.loc[start_date:end_date]["Close"]
    spy_equity = (spy_window / spy_window.iloc[0]) * initial_capital
    spy_equity.name = "spy_equity"

    # 메트릭 계산
    from backtest.metrics import (
        total_return, annualized_return, annualized_volatility,
        sharpe_ratio, max_drawdown, alpha_vs_benchmark,
    )

    summary = {
        "start_date": str(equity_curve.index[0].date()),
        "end_date": str(equity_curve.index[-1].date()),
        "n_rebalances": len(rebalances),
        "initial_capital": initial_capital,
        "final_capital": float(equity_curve.iloc[-1]),
        "total_return": total_return(equity_curve),
        "annualized_return": annualized_return(equity_curve),
        "annualized_volatility": annualized_volatility(equity_curve),
        "sharpe_ratio": sharpe_ratio(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "spy_total_return": total_return(spy_equity),
        "spy_annualized_return": annualized_return(spy_equity),
        "alpha_vs_spy": alpha_vs_benchmark(equity_curve, spy_equity),
    }

    return {
        "equity_curve": equity_curve,
        "spy_equity": spy_equity,
        "rebalances": rebalances,
        "summary": summary,
    }


def _format_summary(s: dict) -> str:
    return (
        f"\n=== 백테스트 결과 ===\n"
        f"기간: {s['start_date']} ~ {s['end_date']}\n"
        f"리밸런싱: {s['n_rebalances']}회\n"
        f"--- 수익 ---\n"
        f"초기: ${s['initial_capital']:,.0f} → 종료: ${s['final_capital']:,.0f}\n"
        f"총 수익률:       {s['total_return']*100:+.2f}%\n"
        f"연환산 수익률:   {s['annualized_return']*100:+.2f}%\n"
        f"연환산 변동성:   {s['annualized_volatility']*100:.2f}%\n"
        f"Sharpe Ratio:    {s['sharpe_ratio']:.2f}\n"
        f"Max Drawdown:    {s['max_drawdown']*100:.2f}%\n"
        f"--- SPY 벤치마크 ---\n"
        f"SPY 총 수익률:   {s['spy_total_return']*100:+.2f}%\n"
        f"SPY 연환산:      {s['spy_annualized_return']*100:+.2f}%\n"
        f"Alpha vs SPY:    {s['alpha_vs_spy']*100:+.2f}% (연환산)\n"
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="추천 시스템 백테스트")
    parser.add_argument("--top-mcap", type=int, default=5000, help="유니버스 시총 상위 N개")
    parser.add_argument("--top-n", type=int, default=10, help="Top N 픽 (포트폴리오 구성)")
    parser.add_argument("--rebalance", default="W-MON",
                        help="리밸런싱 빈도 (pandas freq: W-MON, M, 2W 등)")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--use-cache", action="store_true",
                        help="data/cache의 당일 가격 캐시 사용 (없으면 yf.download 호출)")
    parser.add_argument("--period", default="3y",
                        help="가격 데이터 기간 (1y/2y/3y/5y, 기본 3y)")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="당일 캐시 무시하고 강제 재다운로드 (period 변경 시 필요)")
    args = parser.parse_args()

    print(f"유니버스 로드 (시총 상위 {args.top_mcap}개)...")
    universe = get_universe(top_n_by_mcap=args.top_mcap)
    tickers = universe["ticker"].tolist()
    print(f"  {len(tickers)}개 종목")

    today_iso = date_cls.today().isoformat()
    if args.refresh_cache:
        print(f"강제 재다운로드 ({args.period} 기간)...")
        results = batch_fetch(tickers, period=args.period, use_cache=False)
        price_data = {r["ticker"]: r["price_df"] for r in results}
    elif args.use_cache:
        price_data = _load_cached_prices(tickers, today_iso)
        print(f"캐시 적중: {len(price_data)}/{len(tickers)}개")
        missing = [t for t in tickers if t not in price_data]
        if missing:
            print(f"누락 {len(missing)}개 → yf.download 보충 호출")
            extra = batch_fetch(missing, period=args.period, use_cache=True)
            for r in extra:
                price_data[r["ticker"]] = r["price_df"]
    else:
        print(f"가격 데이터 수집 ({args.period}, yf.download bulk)...")
        results = batch_fetch(tickers, period=args.period, use_cache=True)
        price_data = {r["ticker"]: r["price_df"] for r in results}

    print(f"수집 완료: {len(price_data)}개 가격 데이터")
    if not price_data:
        print("데이터 없음, 종료.")
        return

    # 인덱스 tz 제거
    for t, df in price_data.items():
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
            price_data[t] = df

    # 백테스트 기간: 데이터에 200일 이상 확보된 첫 시점부터 마지막까지
    all_dates = sorted(set().union(*[df.index for df in price_data.values()]))
    if len(all_dates) < 200:
        print(f"전체 거래일 {len(all_dates)}일로는 부족 (200일+ 필요)")
        return
    start_date = all_dates[200]  # 200일 warmup
    end_date = all_dates[-1]
    print(f"백테스트 기간: {start_date.date()} ~ {end_date.date()}")

    print("SPY 벤치마크 다운로드...")
    spy = _get_spy(period="2y")  # 여유있게 2y 받아서 시작일에 맞춤

    print(f"백테스트 실행 (Top {args.top_n}, {args.rebalance} 리밸런싱)...")
    result = run_backtest(
        price_data=price_data,
        spy_df=spy,
        start_date=start_date,
        end_date=end_date,
        top_n=args.top_n,
        rebalance_freq=args.rebalance,
        initial_capital=args.capital,
    )

    print(_format_summary(result["summary"]))

    # 저장
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = BACKTEST_DIR / f"{ts}_summary.json"
    equity_path = BACKTEST_DIR / f"{ts}_equity.parquet"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, default=str)
    eq_df = pd.DataFrame({
        "strategy": result["equity_curve"],
        "spy": result["spy_equity"].reindex(result["equity_curve"].index, method="ffill"),
    })
    eq_df.to_parquet(equity_path)
    print(f"\n저장: {summary_path.name}, {equity_path.name}")

    # 리밸런싱 로그 간단 출력
    print(f"\n--- 리밸런싱 로그 (처음 5개) ---")
    for r in result["rebalances"][:5]:
        print(f"  {r['date'].date()}: {r['tickers']}")
    if len(result["rebalances"]) > 5:
        print(f"  ... 외 {len(result['rebalances']) - 5}회")


if __name__ == "__main__":
    main()
