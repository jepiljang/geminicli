"""
백테스팅 대시보드 (Streamlit).

실행:
    streamlit run backtest/app.py
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (streamlit 실행 시 import 경로 문제 방지)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data.fetcher import fetch_data
from features.technical import add_all_indicators
from features.custom import add_all_custom_features
from strategy.base import MultiFactorStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import summarize, check_kpi


# ======================================================================
# Streamlit 설정
# ======================================================================
st.set_page_config(
    page_title="미국주식 백테스팅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 미국주식 전략 백테스팅")
st.caption("SPY 대비 초과수익 + Win Rate 55% 달성 여부 검증")


# ======================================================================
# 사이드바: 파라미터 입력
# ======================================================================
st.sidebar.header("⚙️ 백테스트 설정")

ticker = st.sidebar.text_input("종목 티커", value="AAPL").strip().upper()
period = st.sidebar.selectbox(
    "기간",
    options=["1y", "2y", "3y", "5y", "10y"],
    index=2,
)
initial_capital = st.sidebar.number_input(
    "초기 자본 ($)",
    min_value=1000,
    max_value=10_000_000,
    value=100_000,
    step=10_000,
)
commission = st.sidebar.slider(
    "수수료 (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
) / 100
slippage = st.sidebar.slider(
    "슬리피지 (%)",
    min_value=0.0,
    max_value=0.5,
    value=0.05,
    step=0.01,
) / 100
position_size = st.sidebar.slider(
    "포지션 사이즈",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 전략 파라미터")

buy_threshold = st.sidebar.slider(
    "매수 임계값",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
)
sell_threshold = st.sidebar.slider(
    "매도 임계값",
    min_value=-1.0,
    max_value=0.0,
    value=-0.30,
    step=0.05,
)

with st.sidebar.expander("팩터 가중치"):
    w_momentum = st.slider("Momentum", 0.0, 0.5, 0.15, 0.05)
    w_trend = st.slider("Trend", 0.0, 0.5, 0.20, 0.05)
    w_volatility = st.slider("Volatility", 0.0, 0.5, 0.10, 0.05)
    w_volume = st.slider("Volume", 0.0, 0.5, 0.10, 0.05)
    w_rs = st.slider("Relative Strength", 0.0, 0.5, 0.15, 0.05)
    w_mr = st.slider("Mean Reversion", 0.0, 0.5, 0.10, 0.05)
    w_regime = st.slider("Market Regime", 0.0, 0.5, 0.10, 0.05)
    w_breakout = st.slider("Breakout", 0.0, 0.5, 0.10, 0.05)

run_button = st.sidebar.button("🚀 백테스트 실행", type="primary", use_container_width=True)


# ======================================================================
# 데이터 로드 (캐싱)
# ======================================================================
@st.cache_data(show_spinner=False)
def load_data(ticker: str, period: str) -> dict:
    """종목 + 벤치마크(SPY) + VIX + 안전자산 데이터 수집."""
    ticker_df = fetch_data(ticker, period, interval="1d")
    spy_df = fetch_data("SPY", period, interval="1d")
    vix_df = fetch_data("^VIX", period, interval="1d")
    gld_df = fetch_data("GLD", period, interval="1d")
    tlt_df = fetch_data("TLT", period, interval="1d")
    uup_df = fetch_data("UUP", period, interval="1d")
    return {
        "ticker": ticker_df,
        "spy": spy_df,
        "vix": vix_df,
        "gld": gld_df,
        "tlt": tlt_df,
        "uup": uup_df,
    }


def build_features(data: dict) -> pd.DataFrame:
    """지표 + 커스텀 피처 생성."""
    df = data["ticker"].copy()
    df = add_all_indicators(df)
    df = add_all_custom_features(
        df,
        benchmark_df=data["spy"],
        vix_df=data["vix"],
        gld_df=data["gld"],
        tlt_df=data["tlt"],
        uup_df=data["uup"],
    )
    return df


def spy_buy_and_hold_equity(spy_df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """SPY 매수 후 보유 전략의 자본 곡선 (벤치마크)."""
    spy_close = spy_df["Close"].dropna()
    if spy_close.empty:
        return pd.Series(dtype=float)
    shares = initial_capital / spy_close.iloc[0]
    return spy_close * shares


# ======================================================================
# 메인 로직
# ======================================================================
if run_button:
    with st.spinner(f"{ticker} 데이터 수집 중..."):
        try:
            data = load_data(ticker, period)
        except Exception as e:
            st.error(f"데이터 수집 실패: {e}")
            st.stop()

    if data["ticker"].empty:
        st.error(f"{ticker} 데이터가 없습니다.")
        st.stop()

    with st.spinner("피처 생성 중..."):
        df_features = build_features(data)

    # 전략 생성
    strategy = MultiFactorStrategy(params={
        "weights": {
            "momentum": w_momentum,
            "trend": w_trend,
            "volatility": w_volatility,
            "volume": w_volume,
            "relative_strength": w_rs,
            "mean_reversion": w_mr,
            "market_regime": w_regime,
            "breakout": w_breakout,
        },
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
    })

    # 백테스트 실행
    with st.spinner("백테스트 실행 중..."):
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            position_size=position_size,
        )
        result = engine.run(df_features, strategy)

    # 벤치마크 자본 곡선
    spy_equity = spy_buy_and_hold_equity(data["spy"], initial_capital)
    spy_equity = spy_equity.reindex(result["equity_curve"].index, method="ffill")

    # 성과 지표
    summary = summarize(
        result["equity_curve"],
        result["trades"],
        benchmark_equity=spy_equity,
    )
    kpi_check = check_kpi(summary)

    # ==================================================================
    # 출력: KPI 요약
    # ==================================================================
    st.header("🎯 KPI 달성 현황")
    kpi_cols = st.columns(5)
    kpi_labels = [
        ("Alpha > 0", "alpha_positive"),
        ("Win Rate > 55%", "win_rate_55"),
        ("Sharpe > 1.0", "sharpe_1"),
        ("MDD > -20%", "mdd_20"),
        ("SPY 초과", "beats_spy"),
    ]
    for col, (label, key) in zip(kpi_cols, kpi_labels):
        check = kpi_check[key]
        icon = "✅" if check["pass"] else "❌"
        actual = check["actual"]
        actual_str = f"{actual:.2%}" if isinstance(actual, (int, float)) else "N/A"
        col.metric(f"{icon} {label}", actual_str, help=f"목표: {check['target']}")

    # ==================================================================
    # 주요 지표
    # ==================================================================
    st.header("📊 성과 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 수익률", f"{summary['total_return']:.2%}")
    c2.metric("연 수익률", f"{summary['annualized_return']:.2%}")
    c3.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
    c4.metric("Max Drawdown", f"{summary['max_drawdown']:.2%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Win Rate", f"{summary['win_rate']:.2%}")
    c6.metric("Profit Factor", f"{summary['profit_factor']:.2f}")
    c7.metric("거래 횟수", f"{summary['num_trades']}")
    c8.metric("평균 보유일", f"{summary['avg_holding_days']:.1f}일")

    if summary.get("benchmark_return") is not None:
        st.subheader("🆚 SPY 벤치마크 비교")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("SPY 연 수익률", f"{summary['benchmark_return']:.2%}")
        b2.metric("초과 수익", f"{summary['excess_return']:.2%}")
        b3.metric("Alpha", f"{summary['alpha_vs_benchmark']:.2%}")
        b4.metric("Beta", f"{summary['beta_vs_benchmark']:.2f}")

    # ==================================================================
    # Equity Curve 차트 (전략 vs SPY)
    # ==================================================================
    st.header("📈 자본 곡선 (전략 vs SPY)")
    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=result["equity_curve"].index,
            y=result["equity_curve"].values,
            name=f"{ticker} 전략",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig_equity.add_trace(
        go.Scatter(
            x=spy_equity.index,
            y=spy_equity.values,
            name="SPY Buy & Hold",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )
    fig_equity.update_layout(
        xaxis_title="날짜",
        yaxis_title="자본 ($)",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # ==================================================================
    # 가격 + 시그널 차트
    # ==================================================================
    st.header("🕯️ 가격 + 매매 시그널")
    signals_df = result["signals_df"]
    buy_signals = signals_df[signals_df["signal"] == 1]
    sell_signals = signals_df[signals_df["signal"] == -1]

    fig_price = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("가격 + 시그널", "Total Score"),
    )

    # 캔들스틱
    fig_price.add_trace(
        go.Candlestick(
            x=signals_df.index,
            open=signals_df["Open"],
            high=signals_df["High"],
            low=signals_df["Low"],
            close=signals_df["Close"],
            name="OHLC",
        ),
        row=1, col=1,
    )

    # SMA
    if "SMA_50" in signals_df.columns:
        fig_price.add_trace(
            go.Scatter(
                x=signals_df.index, y=signals_df["SMA_50"],
                name="SMA 50", line=dict(color="orange", width=1),
            ),
            row=1, col=1,
        )
    if "SMA_200" in signals_df.columns:
        fig_price.add_trace(
            go.Scatter(
                x=signals_df.index, y=signals_df["SMA_200"],
                name="SMA 200", line=dict(color="purple", width=1),
            ),
            row=1, col=1,
        )

    # Buy/Sell 마커
    fig_price.add_trace(
        go.Scatter(
            x=buy_signals.index, y=buy_signals["Close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="green"),
            name="Buy",
        ),
        row=1, col=1,
    )
    fig_price.add_trace(
        go.Scatter(
            x=sell_signals.index, y=sell_signals["Close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="red"),
            name="Sell",
        ),
        row=1, col=1,
    )

    # Total Score
    if "total_score" in signals_df.columns:
        fig_price.add_trace(
            go.Scatter(
                x=signals_df.index, y=signals_df["total_score"],
                name="Total Score",
                line=dict(color="#2ca02c", width=1.5),
            ),
            row=2, col=1,
        )
        fig_price.add_hline(
            y=buy_threshold, line_dash="dash", line_color="green",
            row=2, col=1,
        )
        fig_price.add_hline(
            y=sell_threshold, line_dash="dash", line_color="red",
            row=2, col=1,
        )

    fig_price.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ==================================================================
    # 팩터 스코어 차트
    # ==================================================================
    st.header("🧮 팩터 스코어 분해")
    factor_cols = [c for c in signals_df.columns if c.startswith("score_")]
    if factor_cols:
        fig_factors = go.Figure()
        for col in factor_cols:
            fig_factors.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df[col],
                    name=col.replace("score_", ""),
                    mode="lines",
                )
            )
        fig_factors.update_layout(
            yaxis_title="Score (-1 ~ +1)",
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig_factors, use_container_width=True)

    # ==================================================================
    # 거래 내역
    # ==================================================================
    st.header("📋 거래 내역")
    trades = result["trades"]
    if not trades.empty:
        display_trades = trades.copy()
        display_trades["pnl_pct"] = (display_trades["pnl_pct"] * 100).round(2)
        display_trades["pnl"] = display_trades["pnl"].round(2)
        display_trades["entry_price"] = display_trades["entry_price"].round(2)
        display_trades["exit_price"] = display_trades["exit_price"].round(2)
        st.dataframe(display_trades, use_container_width=True, hide_index=True)
    else:
        st.info("거래가 발생하지 않았습니다. 임계값을 낮춰보세요.")

    # ==================================================================
    # 전체 지표 (expander)
    # ==================================================================
    with st.expander("📈 전체 지표 (raw)"):
        st.json(summary)

else:
    st.info("👈 사이드바에서 설정 후 '백테스트 실행' 버튼을 눌러주세요.")
    st.markdown("""
    ### 사용 방법
    1. **종목 티커** 입력 (예: AAPL, MSFT, NVDA)
    2. **기간** 선택 (1~10년)
    3. **전략 파라미터** 조정 (팩터 가중치, 임계값)
    4. **백테스트 실행** 클릭

    ### KPI 목표
    - Alpha vs SPY > 0 (초과 수익)
    - Win Rate > 55%
    - Sharpe Ratio > 1.0
    - Max Drawdown < -20%
    """)
