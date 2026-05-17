"""scorer.py의 7번째 팩터 통합 회귀 테스트."""

import numpy as np
import pandas as pd
import pytest

from recommendation.exemplar.profile import build_profile_from_df
from recommendation.scorer import RecommendationScorer, build_weights


def test_build_weights_no_exemplar_returns_six_factor():
    w = build_weights(exemplar_weight=0.0)
    assert "exemplar" not in w
    assert set(w.keys()) == {"momentum", "trend", "breakout", "valuation", "growth", "risk"}
    assert sum(w.values()) == pytest.approx(1.0)


def test_build_weights_with_exemplar_scales_six_factors():
    w = build_weights(exemplar_weight=0.20)
    assert w["exemplar"] == pytest.approx(0.20)
    assert w["momentum"] == pytest.approx(0.20 * 0.80)
    assert w["trend"] == pytest.approx(0.20 * 0.80)
    assert sum(w.values()) == pytest.approx(1.0)


def test_score_single_no_profiles_omits_exemplar_keys(rising_ohlcv):
    scorer = RecommendationScorer()
    result = scorer.score_single({
        "ticker": "TEST", "price_df": rising_ohlcv, "info": {},
    })
    assert result is not None
    assert "exemplar_similarity" not in result
    assert "best_match_id" not in result


def test_score_single_with_profiles_adds_exemplar_keys(rising_ohlcv):
    profile = build_profile_from_df(rising_ohlcv)
    scorer = RecommendationScorer(
        weights=build_weights(exemplar_weight=0.20),
        profiles=[("self", profile)],
    )
    result = scorer.score_single({
        "ticker": "TEST", "price_df": rising_ohlcv, "info": {},
    })
    assert result is not None
    # 자기 자신 vs 자기 프로파일 → exemplar_similarity 높아야 함
    assert result["exemplar_similarity"] > 50
    assert result["best_match_id"] == "self"


def test_score_all_normalizes_exemplar_column(rising_ohlcv, flat_ohlcv):
    profile = build_profile_from_df(rising_ohlcv)
    scorer = RecommendationScorer(
        weights=build_weights(exemplar_weight=0.20),
        profiles=[("rally", profile)],
    )
    data = [
        {"ticker": "A", "price_df": rising_ohlcv, "info": {}},
        {"ticker": "B", "price_df": flat_ohlcv, "info": {}},
    ]
    df = scorer.score_all(data)
    assert "exemplar_similarity" in df.columns
    # 백분위 정규화: 두 종목이면 50, 100
    assert set(df["exemplar_similarity"].round(0).tolist()) == {50.0, 100.0}
    # rising 종목이 더 높은 점수
    rising_row = df[df["ticker"] == "A"].iloc[0]
    flat_row = df[df["ticker"] == "B"].iloc[0]
    assert rising_row["exemplar_similarity"] > flat_row["exemplar_similarity"]


def test_zero_exemplar_weight_matches_legacy_total(rising_ohlcv):
    """7번째 가중치=0이면 기존 6팩터 결과와 동일해야 한다 (회귀 방지)."""
    legacy = RecommendationScorer()
    legacy_df = legacy.score_all([
        {"ticker": "A", "price_df": rising_ohlcv, "info": {}},
    ])

    new = RecommendationScorer(weights=build_weights(exemplar_weight=0.0))
    new_df = new.score_all([
        {"ticker": "A", "price_df": rising_ohlcv, "info": {}},
    ])

    assert legacy_df["total_score"].iloc[0] == pytest.approx(new_df["total_score"].iloc[0])
