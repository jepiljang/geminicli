"""
Cross-sectional 종목 랭킹.

아이디어: 지난 7일간 1시간봉 중에서
          (stock_bar_return > spy_bar_return) 인 바 개수를
          센 뒤, 상위 N개 종목 선정.

"SPY보다 더 자주 이긴 종목"이 단기 상대강도를 대변한다는 가정.

Look-ahead bias 방지:
- 랭킹 시점 t 이전의 데이터만 사용
- 반환값은 "지금 t 시점에서 볼 수 있는 랭킹"
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_outperform_score(
    stock_bars: pd.DataFrame,
    spy_bars: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_bars: int = 49,  # 7일 * 7시간/일 ≈ 49
) -> Optional[float]:
    """
    as_of 시점 직전 lookback_bars개의 1h 바 중,
    (stock 수익률 > SPY 수익률) 인 바의 비율 반환.

    Args:
        stock_bars: 종목 1h OHLCV (Close 필요)
        spy_bars: SPY 1h OHLCV (Close 필요)
        as_of: 랭킹 기준 시점 (이 시점 직전까지의 바만 사용)
        lookback_bars: 과거 몇 개 바를 볼지

    Returns:
        outperform 비율 (0.0 ~ 1.0). 데이터 부족 시 None.
    """
    if "Close" not in stock_bars.columns or "Close" not in spy_bars.columns:
        return None

    stock_close = stock_bars["Close"]
    spy_close = spy_bars["Close"]

    # as_of 이전 데이터만 (strict less than으로 look-ahead 방지)
    stock_close = stock_close[stock_close.index < as_of]
    spy_close = spy_close[spy_close.index < as_of]

    if len(stock_close) < lookback_bars + 1 or len(spy_close) < lookback_bars + 1:
        return None

    # 공통 인덱스로 정렬
    stock_close = stock_close.tail(lookback_bars + 1)
    spy_close = spy_close.reindex(stock_close.index, method="ffill")

    if spy_close.isna().any() or stock_close.isna().any():
        return None

    stock_ret = stock_close.pct_change().dropna()
    spy_ret = spy_close.pct_change().dropna()

    common = stock_ret.index.intersection(spy_ret.index)
    if len(common) < lookback_bars * 0.8:
        return None

    wins = (stock_ret.loc[common] > spy_ret.loc[common]).sum()
    total = len(common)
    return float(wins) / float(total)


def rank_stocks(
    stocks_bars: Dict[str, pd.DataFrame],
    spy_bars: pd.DataFrame,
    as_of: pd.Timestamp,
    top_n: int = 5,
    lookback_bars: int = 49,
    reverse: bool = False,
) -> List[str]:
    """
    as_of 시점에서 상위 top_n 종목 티커 리스트 반환.

    Args:
        stocks_bars: {ticker: 1h DataFrame} 딕셔너리
        spy_bars: SPY 1h DataFrame
        as_of: 랭킹 기준 시점 (해당 시점 이전 데이터만 사용)
        top_n: 선정할 종목 수
        lookback_bars: 룩백 윈도우 (1h 바 단위)
        reverse: True면 오름차순 (underperformer 선정 = mean reversion)

    Returns:
        정렬된 상위 티커 리스트
    """
    scores = {}
    for ticker, bars in stocks_bars.items():
        score = compute_outperform_score(bars, spy_bars, as_of, lookback_bars)
        if score is not None:
            scores[ticker] = score

    # 결정론성: 점수 후 티커 알파벳
    if reverse:
        ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
    else:
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ticker for ticker, _ in ranked[:top_n]]
