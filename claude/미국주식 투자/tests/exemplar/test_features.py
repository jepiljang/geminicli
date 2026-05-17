"""features.py 단위 테스트."""

import math

import numpy as np
import pandas as pd
import pytest

from recommendation.exemplar.features import (
    FEATURE_KEYS,
    compute_features_series,
    compute_features_snapshot,
)


def test_feature_keys_is_10():
    assert len(FEATURE_KEYS) == 10
    assert set(FEATURE_KEYS) == {
        "rsi_14", "ret_5d", "ret_20d", "macd_hist_norm", "adx_14",
        "sma20_pos", "sma50_pos", "high_52w_pos", "bb_pos", "volume_ratio",
    }


def test_series_returns_dataframe_with_all_features(rising_ohlcv):
    out = compute_features_series(rising_ohlcv)
    assert isinstance(out, pd.DataFrame)
    assert set(FEATURE_KEYS).issubset(set(out.columns))


def test_series_drops_initial_nan_rows(rising_ohlcv):
    out = compute_features_series(rising_ohlcv)
    # SMA50 needs 50 rows; 첫 49행은 NaN → drop 되어야 함
    assert len(out) < len(rising_ohlcv)
    assert not out.isna().any().any()


def test_snapshot_returns_dict_with_all_features(rising_ohlcv):
    snap = compute_features_snapshot(rising_ohlcv)
    assert isinstance(snap, dict)
    assert set(snap.keys()) == set(FEATURE_KEYS)
    for v in snap.values():
        assert isinstance(v, float)
        assert not math.isnan(v)


def test_snapshot_matches_last_row_of_series(rising_ohlcv):
    series = compute_features_series(rising_ohlcv)
    snap = compute_features_snapshot(rising_ohlcv)
    last = series.iloc[-1]
    for key in FEATURE_KEYS:
        assert snap[key] == pytest.approx(last[key], rel=1e-6)


def test_rising_data_has_high_52w_near_1(rising_ohlcv):
    """우상향 데이터는 마지막에 52주 고가 근처여야 한다."""
    snap = compute_features_snapshot(rising_ohlcv)
    assert snap["high_52w_pos"] > 0.85


def test_series_insufficient_data_raises():
    short_df = pd.DataFrame({
        "Open": [1.0] * 30, "High": [1.0] * 30, "Low": [1.0] * 30,
        "Close": [1.0] * 30, "Volume": [1.0] * 30,
    })
    with pytest.raises(ValueError, match="at least 50"):
        compute_features_series(short_df)
