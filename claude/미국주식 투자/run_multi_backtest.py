"""
다종목 포트폴리오 백테스트 CLI.

전략:
- SPY 일봉 + MultiFactorStrategy → 마켓 on/off 시그널
- 시그널 on인 동안 S&P500 상위 N 종목 (1h 상대강도 랭킹) 보유
- Equal weight, SPY sell 시그널 뜨면 전량 청산

사용:
    python run_multi_backtest.py --period 2y --top-n 5
"""
import argparse
import io
import sys
import time
from pathlib import Path

# Windows cp949 터미널 UTF-8 wrapper
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fetcher import fetch_data
from data.universe import get_sp500_top50
from features.technical import add_all_indicators
from features.custom import add_all_custom_features
from strategy.base import MultiFactorStrategy
from backtest.multi_engine import MultiStockBacktestEngine
from backtest.metrics import summarize, check_kpi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=str, default="2y",
                        help="일봉 기간 (2y 권장, 1h는 내부적으로 730d 제한)")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--top-n", dest="top_n", type=int, default=5)
    parser.add_argument("--buy-th", dest="buy_th", type=float, default=0.15)
    parser.add_argument("--sell-th", dest="sell_th", type=float, default=-0.05)
    parser.add_argument("--lookback-bars", dest="lookback_bars", type=int, default=49,
                        help="랭킹 룩백 (1h 바 개수, 49 ≈ 7일)")
    parser.add_argument("--rebalance-days", dest="rebalance_days", type=int, default=20,
                        help="주기적 리밸런싱 주기 (0=비활성)")
    parser.add_argument("--reverse-rank", dest="reverse_rank", action="store_true",
                        help="역랭킹 (underperformer = mean reversion)")
    parser.add_argument("--max-tickers", dest="max_tickers", type=int, default=50,
                        help="유니버스 크기 제한 (디버깅용)")
    args = parser.parse_args()

    universe = get_sp500_top50()[:args.max_tickers]
    print(f"=== Multi-Stock Backtest ===")
    print(f"  유니버스: {len(universe)}종목, 기간: {args.period}, Top N: {args.top_n}")
    print()

    # ====================================================================
    # [1/6] SPY 일봉 + 보조자산 수집 (MultiFactorStrategy용)
    # ====================================================================
    print("[1/6] SPY 일봉 + 보조자산 수집...")
    t0 = time.time()
    spy_daily = fetch_data("SPY", args.period, "1d")
    vix_df = fetch_data("^VIX", args.period, "1d")
    gld_df = fetch_data("GLD", args.period, "1d")
    tlt_df = fetch_data("TLT", args.period, "1d")
    uup_df = fetch_data("UUP", args.period, "1d")
    print(f"  SPY: {len(spy_daily)} rows  ({time.time()-t0:.1f}s)")
    print()

    # ====================================================================
    # [2/6] SPY 시그널 생성 (MultiFactorStrategy on SPY)
    # ====================================================================
    print("[2/6] SPY 시그널 생성...")
    spy_feat = add_all_indicators(spy_daily.copy())
    spy_feat = add_all_custom_features(
        spy_feat,
        benchmark_df=spy_daily,  # SPY 대비 RS는 1.0
        vix_df=vix_df, gld_df=gld_df, tlt_df=tlt_df, uup_df=uup_df,
    )
    strategy = MultiFactorStrategy(params={
        "buy_threshold": args.buy_th,
        "sell_threshold": args.sell_th,
    })
    spy_signals = strategy.generate_signals(spy_feat)
    sig_counts = spy_signals["signal"].value_counts().to_dict()
    print(f"  SPY 시그널 분포: {sig_counts}")
    print()

    # ====================================================================
    # [3/6] 50종목 일봉 수집
    # ====================================================================
    print(f"[3/6] {len(universe)}종목 일봉 수집...")
    t0 = time.time()
    stocks_daily = {}
    for i, tk in enumerate(universe):
        df = fetch_data(tk, args.period, "1d")
        if not df.empty:
            stocks_daily[tk] = df
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(universe)} 완료...")
    print(f"  성공: {len(stocks_daily)}/{len(universe)}  ({time.time()-t0:.1f}s)")
    print()

    # ====================================================================
    # [4/6] 50종목 1h 봉 수집 (최대 730d = 2y)
    # ====================================================================
    # yfinance 1h 제한: 730일
    hourly_period = "730d" if args.period in ("2y", "3y", "5y", "max") else args.period
    print(f"[4/6] {len(universe)}종목 1h 봉 수집 (period={hourly_period})...")
    t0 = time.time()
    spy_hourly = fetch_data("SPY", hourly_period, "1h")
    print(f"  SPY 1h: {len(spy_hourly)} rows")

    stocks_hourly = {}
    for i, tk in enumerate(universe):
        df = fetch_data(tk, hourly_period, "1h")
        if not df.empty:
            stocks_hourly[tk] = df
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(universe)} 완료...")
    print(f"  성공: {len(stocks_hourly)}/{len(universe)}  ({time.time()-t0:.1f}s)")
    print()

    # ====================================================================
    # [5/6] 다종목 백테스트 실행
    # ====================================================================
    print("[5/6] 다종목 백테스트 실행...")
    engine = MultiStockBacktestEngine(
        initial_capital=args.capital,
        top_n=args.top_n,
        lookback_hourly_bars=args.lookback_bars,
        rebalance_days=args.rebalance_days,
        reverse_rank=args.reverse_rank,
    )
    result = engine.run(
        spy_signals_daily=spy_signals,
        stocks_daily=stocks_daily,
        stocks_hourly=stocks_hourly,
        spy_hourly=spy_hourly,
    )
    print(f"  거래 수: {len(result['trades'])}")
    print(f"  리밸런스 횟수: {len(result['rebalance_log'])}")
    print(f"  종료 자본: ${result['final_capital']:,.2f}")
    print()

    # 리밸런스 로그 출력
    if result["rebalance_log"]:
        print("  리밸런스 이력:")
        for rb in result["rebalance_log"]:
            print(f"    {rb['date'].date()} → {rb['tickers']}")
        print()

    # 거래 내역 요약 (10개만)
    tr = result["trades"]
    if not tr.empty:
        print(f"  거래 내역 (총 {len(tr)}건, 상위 10건):")
        for _, row in tr.head(10).iterrows():
            print(f"    {row['ticker']:6s} {row['entry_date'].date()} → "
                  f"{row['exit_date'].date()} "
                  f"({row['holding_days']}일) {row['pnl_pct']*100:+.2f}%")
        print()

    # ====================================================================
    # [6/6] 성과 지표
    # ====================================================================
    print("[6/6] 성과 지표 계산...")
    spy_close = spy_daily["Close"].dropna()
    spy_shares = args.capital / spy_close.iloc[0]
    spy_equity = (spy_close * spy_shares).reindex(
        result["equity_curve"].index, method="ffill"
    )

    summary = summarize(result["equity_curve"], result["trades"], spy_equity)
    kpi = check_kpi(summary)

    print("\n" + "=" * 60)
    print(f"결과: Multi-Stock Portfolio (Top {args.top_n})")
    print("=" * 60)

    def pct(v):
        return f"{v:.2%}" if isinstance(v, (int, float)) else "N/A"

    def num(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"

    print(f"총 수익률:         {pct(summary['total_return'])}")
    print(f"연 수익률:         {pct(summary['annualized_return'])}")
    print(f"연 변동성:         {pct(summary['annualized_volatility'])}")
    print(f"Sharpe Ratio:      {num(summary['sharpe_ratio'])}")
    print(f"Sortino Ratio:     {num(summary['sortino_ratio'])}")
    print(f"Max Drawdown:      {pct(summary['max_drawdown'])}")
    print(f"Calmar Ratio:      {num(summary['calmar_ratio'])}")
    print(f"거래 수:           {summary['num_trades']}")
    print(f"Win Rate:          {pct(summary['win_rate'])}")
    print(f"Profit Factor:     {num(summary['profit_factor'])}")
    print(f"평균 보유일:       {num(summary['avg_holding_days'])}")
    print()
    print(f"SPY 연수익률:      {pct(summary['benchmark_return'])}")
    print(f"초과 수익:         {pct(summary['excess_return'])}")
    print(f"Alpha:             {pct(summary['alpha_vs_benchmark'])}")
    print(f"Beta:              {num(summary['beta_vs_benchmark'])}")
    print(f"Info Ratio:        {num(summary['information_ratio'])}")

    print("\n" + "=" * 60)
    print("KPI 체크")
    print("=" * 60)
    for name, check in kpi.items():
        icon = "✅" if check["pass"] else "❌"
        actual = check["actual"]
        actual_str = pct(actual) if isinstance(actual, float) else str(actual)
        print(f"  {icon} {name:20s} {check['target']:15s} → {actual_str}")


if __name__ == "__main__":
    main()
