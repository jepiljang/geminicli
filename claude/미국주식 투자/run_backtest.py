"""
CLI 백테스트 실행 스크립트.

사용:
    python run_backtest.py --ticker AAPL --period 3y
"""
import argparse
import io
import sys
from pathlib import Path

# Windows cp949 터미널에서도 한글/이모지 출력 가능하게
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fetcher import fetch_data
from features.technical import add_all_indicators
from features.custom import add_all_custom_features
from strategy.base import MultiFactorStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import summarize, check_kpi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--period", type=str, default="3y")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--buy-th", dest="buy_th", type=float, default=0.30)
    parser.add_argument("--sell-th", dest="sell_th", type=float, default=-0.30)
    args = parser.parse_args()

    print(f"=== Backtest: {args.ticker} ({args.period}) ===\n")

    print("[1/5] 데이터 수집 중...")
    ticker_df = fetch_data(args.ticker, args.period, "1d")
    spy_df = fetch_data("SPY", args.period, "1d")
    vix_df = fetch_data("^VIX", args.period, "1d")
    gld_df = fetch_data("GLD", args.period, "1d")
    tlt_df = fetch_data("TLT", args.period, "1d")
    uup_df = fetch_data("UUP", args.period, "1d")
    print(f"  {args.ticker}: {len(ticker_df)} rows")
    print(f"  SPY: {len(spy_df)} rows\n")

    print("[2/5] 기술적 지표 생성...")
    df = add_all_indicators(ticker_df)
    print(f"  컬럼 수: {len(df.columns)}\n")

    print("[3/5] 커스텀 피처 생성...")
    df = add_all_custom_features(
        df,
        benchmark_df=spy_df,
        vix_df=vix_df,
        gld_df=gld_df,
        tlt_df=tlt_df,
        uup_df=uup_df,
    )
    print(f"  컬럼 수: {len(df.columns)}\n")

    print("[4/5] 백테스트 실행...")
    strategy = MultiFactorStrategy(params={
        "buy_threshold": args.buy_th,
        "sell_threshold": args.sell_th,
    })
    engine = BacktestEngine(initial_capital=args.capital)
    result = engine.run(df, strategy)
    print(f"  거래 수: {len(result['trades'])}")
    print(f"  종료 자본: ${result['final_capital']:,.2f}")

    # 시그널 분포 진단
    sig_df = result["signals_df"]
    sig_counts = sig_df["signal"].value_counts().to_dict()
    print(f"  시그널 분포: {sig_counts}")
    if "total_score" in sig_df.columns:
        score = sig_df["total_score"].dropna()
        print(f"  total_score: min={score.min():.3f}, max={score.max():.3f}, "
              f"mean={score.mean():.3f}, std={score.std():.3f}")
        print(f"  score > {args.buy_th} (buy th): {(score > args.buy_th).sum()}일")
        print(f"  score < {args.sell_th} (sell th): {(score < args.sell_th).sum()}일")

    # 거래 내역 출력
    tr = result["trades"]
    if not tr.empty:
        print("  거래 내역:")
        for _, row in tr.iterrows():
            print(f"    {row['entry_date'].date()} → {row['exit_date'].date()} "
                  f"({row['holding_days']}일) "
                  f"${row['entry_price']:.2f} → ${row['exit_price']:.2f} "
                  f"({row['pnl_pct']*100:+.2f}%)")
    print()

    print("[5/5] 성과 지표 계산...")
    # SPY 벤치마크 (Buy & Hold)
    spy_close = spy_df["Close"].dropna()
    spy_shares = args.capital / spy_close.iloc[0]
    spy_equity = (spy_close * spy_shares).reindex(
        result["equity_curve"].index, method="ffill"
    )

    summary = summarize(result["equity_curve"], result["trades"], spy_equity)
    kpi = check_kpi(summary)

    print("\n" + "=" * 60)
    print(f"결과: {args.ticker} ({args.period})")
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
