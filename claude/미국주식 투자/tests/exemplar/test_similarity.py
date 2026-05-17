"""similarity.py 단위 테스트."""

import pytest

from recommendation.exemplar.features import FEATURE_KEYS
from recommendation.exemplar.similarity import (
    SimilarityResult,
    score_candidate,
)


def _make_profile(means: dict[str, float], std: float = 0.1) -> dict[str, dict[str, float]]:
    """주어진 평균과 균일 std로 프로파일을 만든다."""
    return {k: {"mean": means.get(k, 0.0), "std": std} for k in FEATURE_KEYS}


def _make_snapshot(values: dict[str, float]) -> dict[str, float]:
    return {k: values.get(k, 0.0) for k in FEATURE_KEYS}


def test_identical_snapshot_scores_100():
    profile = _make_profile({k: 0.5 for k in FEATURE_KEYS}, std=0.1)
    snapshot = _make_snapshot({k: 0.5 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [("ex1", profile)])
    assert result.score == pytest.approx(100.0, abs=1e-3)
    assert result.best_match_id == "ex1"


def test_far_snapshot_scores_low():
    profile = _make_profile({k: 0.0 for k in FEATURE_KEYS}, std=0.1)
    # 모든 피처가 모범에서 +10σ 떨어짐
    snapshot = _make_snapshot({k: 1.0 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [("ex1", profile)])
    assert result.score < 5.0


def test_empty_profiles_returns_none():
    snapshot = _make_snapshot({k: 0.5 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [])
    assert result.score is None
    assert result.best_match_id is None
    assert result.breakdown == {}


def test_zero_std_does_not_explode():
    profile = _make_profile({k: 0.0 for k in FEATURE_KEYS}, std=0.0)
    snapshot = _make_snapshot({k: 0.0 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [("ex1", profile)])
    # std=0 + identical mean → 거리 0 → 100
    assert result.score == pytest.approx(100.0, abs=1e-3)


def test_max_similarity_picks_closest_exemplar():
    near_profile = _make_profile({k: 0.5 for k in FEATURE_KEYS}, std=0.1)
    far_profile = _make_profile({k: 5.0 for k in FEATURE_KEYS}, std=0.1)
    snapshot = _make_snapshot({k: 0.5 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [("far", far_profile), ("near", near_profile)])
    assert result.best_match_id == "near"
    assert result.score == pytest.approx(100.0, abs=1e-3)
    assert result.breakdown["near"] > result.breakdown["far"]


def test_breakdown_contains_all_exemplars():
    p1 = _make_profile({k: 0.0 for k in FEATURE_KEYS}, std=0.1)
    p2 = _make_profile({k: 1.0 for k in FEATURE_KEYS}, std=0.1)
    snapshot = _make_snapshot({k: 0.5 for k in FEATURE_KEYS})
    result = score_candidate(snapshot, [("p1", p1), ("p2", p2)])
    assert set(result.breakdown.keys()) == {"p1", "p2"}
