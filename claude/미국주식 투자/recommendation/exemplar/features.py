"""모범 매칭용 10개 기술적 피처 계산.

시계열 (구간 집계용)과 스냅샷 (후보 평가용) 두 가지 API를 제공한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


FEATURE_KEYS: list[str] = [
    "rsi_14",
    "ret_5d",
    "ret_20d",
    "macd_hist_norm",
    "adx_14",
    "sma20_pos",
    "sma50_pos",
    "high_52w_pos",
    "bb_pos",
    "volume_ratio",
]

_MIN_ROWS = 50  # SMA50 / MACD가 안정화되는 최소 길이


def _compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → 10피처 시계열을 계산하고 NaN 행을 그대로 둔다."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = ta.momentum.rsi(close, window=14)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd_hist_norm"] = macd.macd_diff() / close

    out["adx_14"] = ta.trend.adx(high, low, close, window=14)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out["sma20_pos"] = close / sma20 - 1
    out["sma50_pos"] = close / sma50 - 1

    rolling_high = close.rolling(252, min_periods=50).max()
    out["high_52w_pos"] = close / rolling_high

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_mid = bb.bollinger_mavg()
    denom = (bb_upper - bb_mid).replace(0, np.nan)
    out["bb_pos"] = (close - bb_mid) / denom

    vol_sma20 = volume.rolling(20).mean().replace(0, np.nan)
    out["volume_ratio"] = volume / vol_sma20

    return out


def compute_features_series(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → 10피처 시계열 DataFrame. NaN 행 제거.

    Raises:
        ValueError: len(df) < 50 인 경우.
    """
    if len(df) < _MIN_ROWS:
        raise ValueError(f"need at least {_MIN_ROWS} rows of OHLCV, got {len(df)}")
    out = _compute_all(df)
    out = out.dropna(how="any")
    return out


def compute_features_snapshot(df: pd.DataFrame) -> dict[str, float]:
    """OHLCV → 마지막 시점 10피처 dict.

    Raises:
        ValueError: len(df) < 50 인 경우, 또는 마지막 행에 NaN이 있는 경우.
    """
    if len(df) < _MIN_ROWS:
        raise ValueError(f"need at least {_MIN_ROWS} rows of OHLCV, got {len(df)}")
    last = _compute_all(df).iloc[-1]
    if last.isna().any():
        missing = last.index[last.isna()].tolist()
        raise ValueError(f"snapshot has NaN features: {missing}")
    return {k: float(last[k]) for k in FEATURE_KEYS}
