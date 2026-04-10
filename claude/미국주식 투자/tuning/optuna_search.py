"""
Optuna 기반 하이퍼파라미터 튜닝 (다종목 포트폴리오 전략).

핵심:
1. 데이터 한 번만 로드 (캐시 사용) → 모든 trial 재사용
2. SPY 피처도 한 번만 계산 → trial마다 MultiFactorStrategy만 재실행
3. **Train/Test 분할 강제** — Train에서 목적함수 최적화, Test에서 최종 검증
4. 복합 목적함수: alpha + (win_rate-0.5)*0.5 - |mdd|

사용:
    python tuning/optuna_search.py --trials 100
"""
import argparse
import io
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Windows 한글 출력 wrapper
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import pandas as pd

from data.cache import cached_fetch
from data.universe import get_sp500_top50
from features.technical import add_all_indicators
from features.custom import add_all_custom_features
from strategy.base import MultiFactorStrategy
from backtest.multi_engine import MultiStockBacktestEngine
from backtest.metrics import summarize, check_kpi


# =========================================================================
# 데이터 번들 로드 (1회만)
# =========================================================================
def load_data_bundle(period_daily: str = "2y", period_hourly: str = "730d") -> Dict[str, Any]:
    """모든 데이터 캐시 로드 + SPY 피처 사전 계산."""
    print(f"[load] SPY 일봉 + 보조자산 ({period_daily})...")
    t0 = time.time()
    spy_daily = cached_fetch("SPY", period_daily, "1d")
    vix_df = cached_fetch("^VIX", period_daily, "1d")
    gld_df = cached_fetch("GLD", period_daily, "1d")
    tlt_df = cached_fetch("TLT", period_daily, "1d")
    uup_df = cached_fetch("UUP", period_daily, "1d")
    print(f"  완료 ({time.time()-t0:.1f}s)")

    print(f"[load] SPY 피처 계산...")
    t0 = time.time()
    spy_feat = add_all_indicators(spy_daily.copy())
    spy_feat = add_all_custom_features(
        spy_feat,
        benchmark_df=spy_daily,
        vix_df=vix_df, gld_df=gld_df, tlt_df=tlt_df, uup_df=uup_df,
    )
    print(f"  완료 ({time.time()-t0:.1f}s, {len(spy_feat.columns)} cols)")

    universe = get_sp500_top50()
    print(f"[load] {len(universe)}종목 일봉 ({period_daily})...")
    t0 = time.time()
    stocks_daily = {}
    for tk in universe:
        df = cached_fetch(tk, period_daily, "1d")
        if not df.empty:
            stocks_daily[tk] = df
    print(f"  완료 ({time.time()-t0:.1f}s, 성공 {len(stocks_daily)}/{len(universe)})")

    print(f"[load] SPY + {len(universe)}종목 1h ({period_hourly})...")
    t0 = time.time()
    spy_hourly = cached_fetch("SPY", period_hourly, "1h")
    stocks_hourly = {}
    for tk in universe:
        df = cached_fetch(tk, period_hourly, "1h")
        if not df.empty:
            stocks_hourly[tk] = df
    print(f"  완료 ({time.time()-t0:.1f}s, 성공 {len(stocks_hourly)}/{len(universe)})")

    return {
        "spy_daily": spy_daily,
        "spy_feat": spy_feat,
        "spy_hourly": spy_hourly,
        "stocks_daily": stocks_daily,
        "stocks_hourly": stocks_hourly,
    }


# =========================================================================
# Train/Test 분할: 유효 시그널 날짜 기준
# =========================================================================
def compute_split_dates(spy_feat: pd.DataFrame, train_ratio: float = 0.7):
    """
    total_score 계산이 가능한 첫 날짜부터 유효하다고 본 뒤,
    앞 train_ratio를 train, 나머지를 test로 분할.
    """
    # 전체 워밍업 이후 날짜 = 주요 피처가 모두 채워진 날짜
    # 여기서는 'Position_in_52w_Range'가 채워진 날짜를 기준으로 삼음
    if "Position_in_52w_Range" in spy_feat.columns:
        valid = spy_feat["Position_in_52w_Range"].dropna().index
    else:
        valid = spy_feat.index

    if len(valid) < 50:
        raise RuntimeError("유효 날짜 부족 — 더 긴 기간이 필요합니다.")

    split_i = int(len(valid) * train_ratio)
    train_start = valid[0]
    train_end = valid[split_i - 1]
    test_start = valid[split_i]
    test_end = valid[-1]

    return train_start, train_end, test_start, test_end


# =========================================================================
# 단일 backtest 실행 (slice 지원)
# =========================================================================
def run_backtest_slice(
    bundle: Dict[str, Any],
    params: Dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float = 100_000,
) -> Dict[str, Any]:
    """bundle + params → 주어진 기간의 백테스트 요약."""
    # SPY 시그널 생성
    strategy = MultiFactorStrategy(params={
        "buy_threshold": params["buy_th"],
        "sell_threshold": params["sell_th"],
    })
    spy_signals = strategy.generate_signals(bundle["spy_feat"].copy())

    # 기간 슬라이스
    sliced = spy_signals.loc[start:end]
    if sliced.empty:
        return {"summary": {}, "trades": pd.DataFrame(), "score": -10.0}

    engine = MultiStockBacktestEngine(
        initial_capital=initial_capital,
        top_n=params["top_n"],
        lookback_hourly_bars=params["lookback_bars"],
        rebalance_days=params["rebalance_days"],
        reverse_rank=params["reverse_rank"],
    )
    result = engine.run(
        spy_signals_daily=sliced,
        stocks_daily=bundle["stocks_daily"],
        stocks_hourly=bundle["stocks_hourly"],
        spy_hourly=bundle["spy_hourly"],
    )

    # SPY 벤치마크 자본곡선
    spy_close = bundle["spy_daily"]["Close"].dropna().loc[start:end]
    if spy_close.empty:
        return {"summary": {}, "trades": result["trades"], "score": -10.0}

    spy_shares = initial_capital / spy_close.iloc[0]
    spy_equity = (spy_close * spy_shares).reindex(
        result["equity_curve"].index, method="ffill"
    )

    summary = summarize(result["equity_curve"], result["trades"], spy_equity)
    return {"summary": summary, "trades": result["trades"], "equity_curve": result["equity_curve"]}


# =========================================================================
# 목적함수: 복합 점수
# =========================================================================
def composite_score(summary: Dict[str, Any], min_trades: int = 5) -> float:
    """
    목적 점수 = alpha + (win_rate-0.5)*0.5 - |mdd|

    거래 수가 너무 적으면 -10 페널티 (랜덤 탐색 방지).
    """
    if not summary:
        return -10.0
    num_trades = summary.get("num_trades", 0) or 0
    if num_trades < min_trades:
        return -10.0

    alpha = summary.get("alpha_vs_benchmark") or 0.0
    wr = summary.get("win_rate") or 0.0
    mdd = summary.get("max_drawdown") or 0.0

    return float(alpha) + (float(wr) - 0.5) * 0.5 - abs(float(mdd))


# =========================================================================
# Optuna objective
# =========================================================================
def make_objective(bundle: Dict[str, Any], train_start, train_end):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "buy_th": trial.suggest_float("buy_th", 0.0, 0.30),
            "sell_th": trial.suggest_float("sell_th", -0.30, 0.0),
            "top_n": trial.suggest_int("top_n", 3, 15),
            "rebalance_days": trial.suggest_int("rebalance_days", 5, 30),
            "lookback_bars": trial.suggest_int("lookback_bars", 20, 100),
            "reverse_rank": trial.suggest_categorical("reverse_rank", [True, False]),
        }
        try:
            res = run_backtest_slice(bundle, params, train_start, train_end)
            return composite_score(res["summary"])
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return -10.0

    return objective


# =========================================================================
# 메인
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--period", type=str, default="2y")
    parser.add_argument("--train-ratio", dest="train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Optuna 튜닝 시작 ===\n")
    bundle = load_data_bundle(args.period)

    train_start, train_end, test_start, test_end = compute_split_dates(
        bundle["spy_feat"], train_ratio=args.train_ratio
    )
    print(f"\n[split]")
    print(f"  Train: {train_start.date()} ~ {train_end.date()}")
    print(f"  Test:  {test_start.date()} ~ {test_end.date()}")
    print()

    # Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # optuna 로그는 WARNING만
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"[optuna] {args.trials} trials 실행 중...")
    t0 = time.time()

    def _cb(study, trial):
        if trial.number % 10 == 0 or trial.number == args.trials - 1:
            best = study.best_value
            elapsed = time.time() - t0
            print(f"  trial {trial.number+1}/{args.trials}  "
                  f"best={best:.4f}  elapsed={elapsed:.0f}s")

    study.optimize(
        make_objective(bundle, train_start, train_end),
        n_trials=args.trials,
        callbacks=[_cb],
        show_progress_bar=False,
    )

    print(f"\n[결과] 총 {time.time()-t0:.0f}s 소요")
    print(f"최고 점수 (train): {study.best_value:.4f}")
    print(f"최고 파라미터:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print()

    # Train 성능 상세
    print("=" * 60)
    print("Train 성능 (최적 파라미터)")
    print("=" * 60)
    train_res = run_backtest_slice(bundle, study.best_params, train_start, train_end)
    _print_summary(train_res["summary"])

    # Test 성능 (out-of-sample 진실의 순간)
    print("\n" + "=" * 60)
    print("Test 성능 (out-of-sample) ← 진짜 성과")
    print("=" * 60)
    test_res = run_backtest_slice(bundle, study.best_params, test_start, test_end)
    _print_summary(test_res["summary"])

    # KPI 비교
    print("\n" + "=" * 60)
    print("KPI 비교 (Train vs Test)")
    print("=" * 60)
    train_kpi = check_kpi(train_res["summary"]) if train_res["summary"] else {}
    test_kpi = check_kpi(test_res["summary"]) if test_res["summary"] else {}
    for key in ["alpha_positive", "win_rate_55", "sharpe_1", "mdd_20", "beats_spy"]:
        t_pass = "✅" if train_kpi.get(key, {}).get("pass") else "❌"
        te_pass = "✅" if test_kpi.get(key, {}).get("pass") else "❌"
        print(f"  {key:20s}  train {t_pass}   test {te_pass}")


def _print_summary(summary: Dict[str, Any]):
    if not summary:
        print("  (데이터 없음)")
        return

    def pct(v):
        return f"{v:.2%}" if isinstance(v, (int, float)) else "N/A"

    def num(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"

    print(f"  총 수익률:    {pct(summary.get('total_return'))}")
    print(f"  연 수익률:    {pct(summary.get('annualized_return'))}")
    print(f"  Sharpe:       {num(summary.get('sharpe_ratio'))}")
    print(f"  MDD:          {pct(summary.get('max_drawdown'))}")
    print(f"  Win Rate:     {pct(summary.get('win_rate'))}")
    print(f"  거래 수:      {summary.get('num_trades', 0)}")
    print(f"  Alpha vs SPY: {pct(summary.get('alpha_vs_benchmark'))}")
    print(f"  SPY 수익률:   {pct(summary.get('benchmark_return'))}")


if __name__ == "__main__":
    main()
