"""
백테스팅 성과 지표 계산 모듈.

프로젝트 KPI:
- Alpha vs SPY > 0
- Win Rate > 55%
- Sharpe Ratio > 1.0
- Max Drawdown < -20%
- 연간 수익률 > SPY 연평균

모든 함수는 순수 함수 (입력 → 출력).
"""
import numpy as np
import pandas as pd
from typing import Optional


# ----------------------------------------------------------------------
# 수익률 지표
# ----------------------------------------------------------------------

def total_return(equity_curve: pd.Series) -> float:
    """누적 수익률. (마지막 자본 / 초기 자본) - 1"""
    if equity_curve.empty or equity_curve.iloc[0] == 0:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR (연복리 수익률)."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    n_periods = len(equity_curve) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    final_ratio = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if final_ratio <= 0:
        return -1.0
    return float(final_ratio ** (1 / years) - 1)


def annualized_volatility(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """연환산 변동성 (일별 수익률 표준편차 * √252)."""
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


# ----------------------------------------------------------------------
# 리스크 지표
# ----------------------------------------------------------------------

def sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Sharpe Ratio. (연수익률 - rf) / 연변동성."""
    returns = equity_curve.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    return float(excess.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Sortino Ratio. Sharpe와 비슷하지만 하방 편차만 사용."""
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return 0.0
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    downside = returns[returns < daily_rf]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    최대 낙폭. 음수 반환 (예: -0.25 = -25%).
    누적 최고점 대비 현재 자본의 하락률 중 가장 큰 값.
    """
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar Ratio. 연수익률 / |MDD|."""
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    ann_ret = annualized_return(equity_curve, periods_per_year)
    return float(ann_ret / abs(mdd))


# ----------------------------------------------------------------------
# 거래 지표 (trades DataFrame 사용)
# ----------------------------------------------------------------------

def win_rate(trades: pd.DataFrame) -> float:
    """승률. 수익 거래 / 전체 거래."""
    if trades.empty or 'pnl' not in trades.columns:
        return 0.0
    wins = (trades['pnl'] > 0).sum()
    return float(wins / len(trades))


def profit_factor(trades: pd.DataFrame) -> float:
    """Profit Factor. 총 이익 / |총 손실|. 손실 0이면 inf 대신 0 반환."""
    if trades.empty or 'pnl' not in trades.columns:
        return 0.0
    gross_profit = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(trades.loc[trades['pnl'] < 0, 'pnl'].sum())
    if gross_loss == 0:
        return 0.0 if gross_profit == 0 else float('inf')
    return float(gross_profit / gross_loss)


def avg_win_loss_ratio(trades: pd.DataFrame) -> float:
    """평균 이익 / 평균 손실 절대값."""
    if trades.empty or 'pnl' not in trades.columns:
        return 0.0
    wins = trades.loc[trades['pnl'] > 0, 'pnl']
    losses = trades.loc[trades['pnl'] < 0, 'pnl']
    if wins.empty or losses.empty:
        return 0.0
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    if avg_loss == 0:
        return 0.0
    return float(avg_win / avg_loss)


def avg_holding_days(trades: pd.DataFrame) -> float:
    """평균 보유 일수."""
    if trades.empty or 'holding_days' not in trades.columns:
        return 0.0
    return float(trades['holding_days'].mean())


# ----------------------------------------------------------------------
# 벤치마크 비교
# ----------------------------------------------------------------------

def _align_series(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """두 Series를 공통 인덱스로 정렬."""
    df = pd.concat([a, b], axis=1, join='inner').dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return df.iloc[:, 0], df.iloc[:, 1]


def beta_vs_benchmark(equity_curve: pd.Series, benchmark_equity: pd.Series) -> float:
    """
    Beta. 전략 수익률과 벤치마크 수익률의 공분산 / 벤치마크 분산.
    """
    s, b = _align_series(equity_curve, benchmark_equity)
    if len(s) < 2:
        return 0.0
    strat_ret = s.pct_change().dropna()
    bench_ret = b.pct_change().dropna()
    s_ret, b_ret = _align_series(strat_ret, bench_ret)
    if len(s_ret) < 2 or b_ret.var() == 0:
        return 0.0
    cov = np.cov(s_ret, b_ret, ddof=0)[0, 1]
    return float(cov / b_ret.var())


def alpha_vs_benchmark(
    equity_curve: pd.Series,
    benchmark_equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Jensen's Alpha (CAPM 기반).
    alpha = strategy_annual_return - (rf + beta * (bench_annual_return - rf))
    """
    s, b = _align_series(equity_curve, benchmark_equity)
    if len(s) < 2:
        return 0.0
    strat_ann = annualized_return(s, periods_per_year)
    bench_ann = annualized_return(b, periods_per_year)
    beta = beta_vs_benchmark(s, b)
    return float(strat_ann - (risk_free_rate + beta * (bench_ann - risk_free_rate)))


def excess_return(
    equity_curve: pd.Series,
    benchmark_equity: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """초과 수익 (단순 차이). strategy_ann_return - bench_ann_return."""
    s, b = _align_series(equity_curve, benchmark_equity)
    if len(s) < 2:
        return 0.0
    return float(
        annualized_return(s, periods_per_year) - annualized_return(b, periods_per_year)
    )


def information_ratio(
    equity_curve: pd.Series,
    benchmark_equity: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Information Ratio. 초과 수익의 평균 / 초과 수익의 표준편차 (tracking error).
    """
    s, b = _align_series(equity_curve, benchmark_equity)
    if len(s) < 2:
        return 0.0
    strat_ret = s.pct_change().dropna()
    bench_ret = b.pct_change().dropna()
    s_ret, b_ret = _align_series(strat_ret, bench_ret)
    if len(s_ret) < 2:
        return 0.0
    active = s_ret - b_ret
    if active.std() == 0:
        return 0.0
    return float(active.mean() / active.std() * np.sqrt(periods_per_year))


# ----------------------------------------------------------------------
# 통합 요약
# ----------------------------------------------------------------------

def summarize(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    benchmark_equity: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    모든 성과 지표를 한번에 계산.

    Args:
        equity_curve: 전략의 자본 곡선 (pd.Series, DatetimeIndex)
        trades: 거래 내역 (engine.run 결과의 'trades' DataFrame)
        benchmark_equity: 벤치마크(SPY) 자본 곡선. None이면 벤치마크 지표는 None
        risk_free_rate: 무위험 수익률 (연환산, 예: 0.04 = 4%)
        periods_per_year: 연환산 기간 (일봉=252)

    Returns:
        모든 지표를 담은 dict
    """
    result = {
        # 수익률
        'total_return': total_return(equity_curve),
        'annualized_return': annualized_return(equity_curve, periods_per_year),
        'annualized_volatility': annualized_volatility(equity_curve, periods_per_year),
        # 리스크
        'sharpe_ratio': sharpe_ratio(equity_curve, risk_free_rate, periods_per_year),
        'sortino_ratio': sortino_ratio(equity_curve, risk_free_rate, periods_per_year),
        'max_drawdown': max_drawdown(equity_curve),
        'calmar_ratio': calmar_ratio(equity_curve, periods_per_year),
        # 거래
        'num_trades': int(len(trades)),
        'win_rate': win_rate(trades),
        'profit_factor': profit_factor(trades),
        'avg_win_loss_ratio': avg_win_loss_ratio(trades),
        'avg_holding_days': avg_holding_days(trades),
        # 벤치마크
        'alpha_vs_benchmark': None,
        'beta_vs_benchmark': None,
        'excess_return': None,
        'information_ratio': None,
        'benchmark_return': None,
    }

    if benchmark_equity is not None and not benchmark_equity.empty:
        result['alpha_vs_benchmark'] = alpha_vs_benchmark(
            equity_curve, benchmark_equity, risk_free_rate, periods_per_year
        )
        result['beta_vs_benchmark'] = beta_vs_benchmark(equity_curve, benchmark_equity)
        result['excess_return'] = excess_return(
            equity_curve, benchmark_equity, periods_per_year
        )
        result['information_ratio'] = information_ratio(
            equity_curve, benchmark_equity, periods_per_year
        )
        result['benchmark_return'] = annualized_return(benchmark_equity, periods_per_year)

    return result


def check_kpi(summary: dict) -> dict:
    """
    KPI 달성 여부 체크.

    Returns:
        각 KPI별 pass/fail + 현재값
    """
    return {
        'alpha_positive': {
            'target': '> 0',
            'actual': summary.get('alpha_vs_benchmark'),
            'pass': (summary.get('alpha_vs_benchmark') or 0) > 0,
        },
        'win_rate_55': {
            'target': '> 0.55',
            'actual': summary.get('win_rate'),
            'pass': summary.get('win_rate', 0) > 0.55,
        },
        'sharpe_1': {
            'target': '> 1.0',
            'actual': summary.get('sharpe_ratio'),
            'pass': summary.get('sharpe_ratio', 0) > 1.0,
        },
        'mdd_20': {
            'target': '> -0.20',
            'actual': summary.get('max_drawdown'),
            'pass': summary.get('max_drawdown', -1) > -0.20,
        },
        'beats_spy': {
            'target': '전략 > SPY',
            'actual': summary.get('excess_return'),
            'pass': (summary.get('excess_return') or 0) > 0,
        },
    }
