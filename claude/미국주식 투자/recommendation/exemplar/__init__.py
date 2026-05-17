"""모범 종목 패턴 매칭 (7번째 팩터) 서브패키지."""

from recommendation.exemplar.features import (
    FEATURE_KEYS,
    compute_features_series,
    compute_features_snapshot,
)
from recommendation.exemplar.library import (
    Exemplar,
    ExemplarLibrary,
    make_exemplar_id,
)
from recommendation.exemplar.profile import (
    build_profile,
    build_profile_from_df,
    load_profile,
    save_profile,
)
from recommendation.exemplar.similarity import (
    SimilarityResult,
    score_candidate,
)

__all__ = [
    "FEATURE_KEYS",
    "compute_features_series",
    "compute_features_snapshot",
    "Exemplar",
    "ExemplarLibrary",
    "make_exemplar_id",
    "build_profile",
    "build_profile_from_df",
    "load_profile",
    "save_profile",
    "SimilarityResult",
    "score_candidate",
]
