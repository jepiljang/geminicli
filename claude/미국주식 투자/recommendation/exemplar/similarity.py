"""후보 종목 vs 모범 프로파일 유사도 계산.

알고리즘:
  z_i = |x_i - μ_i| / max(σ_i, ε)
  d   = mean(z_i)
  similarity = 100 * exp(-d / 2)
라이브러리 점수는 활성 모범들 중 최댓값.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from recommendation.exemplar.features import FEATURE_KEYS

_EPS = 1e-6


@dataclass
class SimilarityResult:
    score: float | None
    best_match_id: str | None
    breakdown: dict[str, float] = field(default_factory=dict)


def _score_one(snapshot: dict[str, float], profile: dict[str, dict[str, float]]) -> float:
    z_sum = 0.0
    for k in FEATURE_KEYS:
        mu = profile[k]["mean"]
        sigma = profile[k]["std"]
        denom = sigma if sigma > _EPS else _EPS
        z_sum += abs(snapshot[k] - mu) / denom
    d = z_sum / len(FEATURE_KEYS)
    return 100.0 * math.exp(-d / 2.0)


def score_candidate(
    snapshot: dict[str, float],
    profiles: list[tuple[str, dict[str, dict[str, float]]]],
) -> SimilarityResult:
    """후보 스냅샷 + (id, profile) 리스트 → SimilarityResult."""
    if not profiles:
        return SimilarityResult(score=None, best_match_id=None, breakdown={})

    breakdown: dict[str, float] = {}
    for exemplar_id, profile in profiles:
        breakdown[exemplar_id] = _score_one(snapshot, profile)

    best_id = max(breakdown, key=breakdown.get)
    return SimilarityResult(
        score=breakdown[best_id],
        best_match_id=best_id,
        breakdown=breakdown,
    )
