"""
다종목 포트폴리오 백테스트 엔진.

전략 구조:
1. SPY 일봉에 MultiFactorStrategy 적용 → 마켓 on/off 시그널
2. SPY 시그널 발생 시: S&P500 유니버스 중 상위 N 종목 랭킹
   - 랭킹 기준: 지난 7일 1h 바 중 (stock_return > spy_return) 바 개수
3. Equal-weight 매수, SPY exit 시그널 뜨면 전량 청산

Look-ahead bias 방지:
- T+1 체결: 시그널은 day t 기준, 매수/매도는 day t+1 open
- 랭킹도 day t+1 open 이전 데이터만 사용 (1h 바 strict < day t+1 midnight)
"""
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from strategy.cross_sectional import rank_stocks


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 인덱스 tz 제거 (비교 안전성)."""
    if df is None or df.empty:
        return df
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


class MultiStockBacktestEngine:
    """
    SPY 마켓 타이밍 + S&P500 상위 N 종목 equal-weight 포트폴리오.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        top_n: int = 5,
        lookback_hourly_bars: int = 49,
        rebalance_days: int = 0,  # 0 = disabled, >0 = force rebalance every N days
        reverse_rank: bool = False,
    ):
        self.initial_capital = float(initial_capital)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.top_n = int(top_n)
        self.lookback_hourly_bars = int(lookback_hourly_bars)
        self.rebalance_days = int(rebalance_days)
        self.reverse_rank = bool(reverse_rank)

    def _apply_slippage(self, price: float, side: str) -> float:
        if side == "buy":
            return price * (1 + self.slippage)
        return price * (1 - self.slippage)

    def run(
        self,
        spy_signals_daily: pd.DataFrame,
        stocks_daily: Dict[str, pd.DataFrame],
        stocks_hourly: Dict[str, pd.DataFrame],
        spy_hourly: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Args:
            spy_signals_daily: SPY 일봉 DataFrame. 반드시 'signal' 컬럼 포함.
            stocks_daily: {ticker: 일봉 OHLCV} — 체결 가격용
            stocks_hourly: {ticker: 1h OHLCV} — 랭킹용
            spy_hourly: SPY 1h OHLCV — 랭킹 기준

        Returns:
            trades, equity_curve, final_capital, rebalance_log
        """
        if "signal" not in spy_signals_daily.columns:
            raise ValueError("spy_signals_daily must contain 'signal' column")

        # tz 정규화
        spy_signals_daily = _strip_tz(spy_signals_daily)
        spy_hourly = _strip_tz(spy_hourly)
        stocks_daily = {t: _strip_tz(df) for t, df in stocks_daily.items()}
        stocks_hourly = {t: _strip_tz(df) for t, df in stocks_hourly.items()}

        # T+1 체결
        exec_signal = (
            spy_signals_daily["signal"].fillna(0).astype(int).shift(1).fillna(0).astype(int)
        )
        dates = spy_signals_daily.index

        cash = self.initial_capital
        # positions: {ticker: {'shares': int, 'entry_price': float, 'entry_date': Timestamp}}
        positions: Dict[str, Dict[str, Any]] = {}

        trades: List[Dict[str, Any]] = []
        equity_history = np.zeros(len(dates), dtype=float)
        rebalance_log: List[Dict[str, Any]] = []

        def current_equity(day_idx: int) -> float:
            """day_idx일의 Close 가격 기준 포트폴리오 평가액."""
            total = cash
            for tk, pos in positions.items():
                df = stocks_daily.get(tk)
                if df is None or df.empty:
                    continue
                # 해당 날짜의 종가 (없으면 직전 종가 ffill)
                sub = df.loc[:dates[day_idx]]
                if sub.empty:
                    continue
                total += pos["shares"] * sub["Close"].iloc[-1]
            return total

        def close_all(day_idx: int, reason: str):
            nonlocal cash
            date = dates[day_idx]
            for tk in list(positions.keys()):
                pos = positions[tk]
                df = stocks_daily.get(tk)
                if df is None or date not in df.index:
                    # 해당 날짜 데이터 없으면 가장 가까운 직전 Close로 청산
                    if df is not None and not df.empty:
                        sub = df.loc[:date]
                        if not sub.empty:
                            exit_px_raw = float(sub["Close"].iloc[-1])
                        else:
                            continue
                    else:
                        continue
                else:
                    exit_px_raw = float(df.loc[date, "Open"])
                    if np.isnan(exit_px_raw):
                        exit_px_raw = float(df.loc[date, "Close"])

                exit_px = self._apply_slippage(exit_px_raw, "sell")
                proceeds = exit_px * pos["shares"]
                commission_cost = proceeds * self.commission
                cash += proceeds - commission_cost

                pnl = (exit_px - pos["entry_price"]) * pos["shares"] - commission_cost
                pnl_pct = (exit_px - pos["entry_price"]) / pos["entry_price"]
                holding_days = (date - pos["entry_date"]).days

                trades.append({
                    "ticker": tk,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "exit_date": date,
                    "exit_price": exit_px,
                    "shares": pos["shares"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "holding_days": holding_days,
                    "exit_reason": reason,
                })
                del positions[tk]

        def enter_topn(day_idx: int):
            nonlocal cash
            date = dates[day_idx]
            ranked = rank_stocks(
                stocks_hourly,
                spy_hourly,
                as_of=date,
                top_n=self.top_n,
                lookback_bars=self.lookback_hourly_bars,
                reverse=self.reverse_rank,
            )
            if not ranked:
                return

            # 각 종목의 day_idx open 가격 확인
            tradable = []
            for tk in ranked:
                df = stocks_daily.get(tk)
                if df is None or date not in df.index:
                    continue
                open_px = float(df.loc[date, "Open"])
                if np.isnan(open_px):
                    continue
                tradable.append((tk, open_px))

            if not tradable:
                return

            per_ticker_cash = cash / len(tradable)

            for tk, open_px_raw in tradable:
                buy_px = self._apply_slippage(open_px_raw, "buy")
                max_shares = int(per_ticker_cash / (buy_px * (1 + self.commission)))
                if max_shares <= 0:
                    continue
                cost = max_shares * buy_px
                commission_cost = cost * self.commission
                cash -= (cost + commission_cost)
                positions[tk] = {
                    "shares": max_shares,
                    "entry_price": buy_px,
                    "entry_date": date,
                }

            rebalance_log.append({
                "date": date,
                "tickers": [tk for tk, _ in tradable],
            })

        # 메인 루프
        last_rebalance_idx = -10**9
        for i in range(len(dates)):
            sig = int(exec_signal.iloc[i])

            if sig == -1 and positions:
                close_all(i, reason="spy_sell_signal")

            elif sig == 1 and not positions:
                enter_topn(i)
                last_rebalance_idx = i

            # 주기적 리밸런싱 (롱 포지션 유지 중 + 일정 간격 경과)
            elif (
                self.rebalance_days > 0
                and positions
                and sig != -1
                and (i - last_rebalance_idx) >= self.rebalance_days
            ):
                close_all(i, reason="periodic_rebalance")
                enter_topn(i)
                last_rebalance_idx = i

            equity_history[i] = current_equity(i)

        # 종료 시점에 잔여 포지션 있으면 마지막 Close 기준으로 평가만 (청산 기록은 안 남김)
        equity_curve = pd.Series(equity_history, index=dates, name="equity")

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
            trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

        return {
            "trades": trades_df,
            "equity_curve": equity_curve,
            "final_capital": float(equity_curve.iloc[-1]),
            "rebalance_log": rebalance_log,
        }
