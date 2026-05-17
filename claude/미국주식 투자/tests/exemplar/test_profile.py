"""profile.py 단위 테스트."""

import pandas as pd
import pytest

from recommendation.exemplar.features import FEATURE_KEYS
from recommendation.exemplar.profile import (
    build_profile_from_df,
    load_profile,
    save_profile,
)


def test_build_profile_from_df_returns_mean_std_dict(rising_ohlcv):
    profile = build_profile_from_df(rising_ohlcv)
    assert set(profile.keys()) == set(FEATURE_KEYS)
    for key, val in profile.items():
        assert "mean" in val and "std" in val
        assert isinstance(val["mean"], float)
        assert isinstance(val["std"], float)
        assert val["std"] >= 0


def test_build_profile_rejects_short_data():
    short_df = pd.DataFrame({
        "Open": [1.0] * 30, "High": [1.0] * 30, "Low": [1.0] * 30,
        "Close": [1.0] * 30, "Volume": [1.0] * 30,
    })
    with pytest.raises(ValueError):
        build_profile_from_df(short_df)


def test_save_and_load_roundtrip(tmp_path, rising_ohlcv):
    profile = build_profile_from_df(rising_ohlcv)
    path = tmp_path / "test.parquet"
    save_profile(path, profile)
    assert path.exists()

    loaded = load_profile(path)
    assert set(loaded.keys()) == set(profile.keys())
    for key in profile:
        assert loaded[key]["mean"] == pytest.approx(profile[key]["mean"])
        assert loaded[key]["std"] == pytest.approx(profile[key]["std"])


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "nope.parquet")
