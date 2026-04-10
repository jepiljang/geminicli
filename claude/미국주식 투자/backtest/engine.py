"""
백테스팅 엔진 (단일 종목).

설계 원칙:
- **T+1 진입**: day t의 시그널 → day t+1의 Open 가격으로 거래 (look-ahead 방지)
- 벡터화 + 이벤트 기반 혼합: 진입/청산 시점만 순회
- 분할 매매 없음 (all-in / all-out)
- 공매도 없음 (signal=-1은 청산만)
- 수수료/슬리피지는 거래금액 기준으로 차감
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

from strategy.base import BaseStrategy


class BacktestEngine:
    """
    단일 종목 백테스팅 엔진.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission: float = 0.001,   # 0.1% per side
        slippage: float = 0.0005,    # 0.05% per side
        position_size: float = 1.0,  # 자본 대비 포지션 비율
    ):
        """
        Args:
            initial_capital: 초기 자본금
            commission: 매수/매도 시 각각 차감되는 수수료 비율 (0.001 = 0.1%)
            slippage: 매수/매도 시 각각 가격에 반영되는 슬리피지 비율
            position_size: 신호 발생 시 사용할 자본 비율 (1.0 = 풀 베팅)
        """
        self.initial_capital = float(initial_capital)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.position_size = float(position_size)

    def _apply_slippage(self, price: float, side: str) -> float:
        """슬리피지를 가격에 반영 (buy=+, sell=-)."""
        if side == 'buy':
            return price * (1 + self.slippage)
        elif side == 'sell':
            return price * (1 - self.slippage)
        raise ValueError(f"Invalid side: {side}")

    def run(self, df: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """
        백테스트 실행.

        Args:
            df: OHLCV + 피처 포함 DataFrame (DatetimeIndex 필요)
            strategy: BaseStrategy 인스턴스

        Returns:
            {
                'trades': pd.DataFrame,       # 개별 거래 내역
                'equity_curve': pd.Series,    # 일별 자본 곡선
                'signals_df': pd.DataFrame,   # 시그널 포함 원본 df
                'final_capital': float,       # 종료 자본금
            }
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex")
        if 'Open' not in df.columns or 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Open' and 'Close' columns")

        # 1. 시그널 생성
        signals_df = strategy.generate_signals(df.copy())
        if 'signal' not in signals_df.columns:
            raise ValueError("Strategy must return a DataFrame with 'signal' column")
        signals_df['signal'] = signals_df['signal'].fillna(0).astype(int)

        # 2. T+1 진입을 위해 signal을 한 칸 shift
        # day t에 발생한 시그널 → day t+1에 체결
        exec_signal = signals_df['signal'].shift(1).fillna(0).astype(int)

        # 3. 이벤트 기반 순회 (진입/청산만)
        opens = signals_df['Open'].values
        closes = signals_df['Close'].values
        dates = signals_df.index

        cash = self.initial_capital
        shares = 0
        entry_price = 0.0
        entry_date = None

        trades = []
        # 일별 보유 주식 수 기록 → equity curve 계산에 사용
        shares_history = np.zeros(len(signals_df), dtype=float)
        cash_history = np.full(len(signals_df), np.nan, dtype=float)

        for i in range(len(signals_df)):
            sig = int(exec_signal.iloc[i])
            price_open = opens[i]

            # 시그널이 NaN이거나 가격이 NaN이면 스킵
            if np.isnan(price_open):
                shares_history[i] = shares
                cash_history[i] = cash
                continue

            # 청산 (sell signal, 포지션 보유 중)
            if sig == -1 and shares > 0:
                exit_price = self._apply_slippage(price_open, 'sell')
                proceeds = exit_price * shares
                commission_cost = proceeds * self.commission
                cash += proceeds - commission_cost

                pnl = (exit_price - entry_price) * shares - commission_cost
                pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                holding_days = (dates[i] - entry_date).days

                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': dates[i],
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'holding_days': holding_days,
                })
                shares = 0
                entry_price = 0.0
                entry_date = None

            # 진입 (buy signal, 포지션 없음)
            elif sig == 1 and shares == 0:
                buy_price = self._apply_slippage(price_open, 'buy')
                # 수수료 포함해서 살 수 있는 최대 수량 계산
                # cash * position_size = shares * buy_price * (1 + commission)
                capital_for_trade = cash * self.position_size
                max_shares = int(capital_for_trade / (buy_price * (1 + self.commission)))

                if max_shares > 0:
                    cost = max_shares * buy_price
                    commission_cost = cost * self.commission
                    cash -= (cost + commission_cost)
                    shares = max_shares
                    entry_price = buy_price
                    entry_date = dates[i]

            shares_history[i] = shares
            cash_history[i] = cash

        # 마지막에 포지션 남아있으면 종가로 청산 (미실현 → 실현)
        # 단, trades에는 기록만 하지 않고 equity에만 반영
        # → 여기서는 equity_curve에 마지막 종가로 평가

        # 4. Equity curve 계산 (벡터화)
        # 매일: cash + shares * Close
        equity_curve = pd.Series(
            cash_history + shares_history * closes,
            index=signals_df.index,
            name='equity',
        )

        # 5. 결과 조립
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

        final_capital = float(equity_curve.iloc[-1])

        return {
            'trades': trades_df,
            'equity_curve': equity_curve,
            'signals_df': signals_df,
            'final_capital': final_capital,
        }
