"""테스트용 합성 OHLCV 데이터 fixture."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rising_ohlcv() -> pd.DataFrame:
    """300일 우상향 가격 + 약한 변동성 OHLCV. 모범 구간 시뮬레이션용."""
    rng = np.random.default_rng(42)
    n = 300
    # 일평균 +0.2% drift, 1.2% daily vol
    returns = rng.normal(0.002, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """300일 횡보 OHLCV. 모범과 다른 패턴 시뮬레이션용."""
    rng = np.random.default_rng(7)
    n = 300
    # drift=0
    returns = rng.normal(0.0, 0.008, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(800_000, 2_000_000, n).astype(float)

    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
