"""Ranking algorithms. This package is the stable in-process API of the ranking service: the retrieval service imports
`fuse_results` from here (no HTTP hop), so the names and signatures below must not change."""
from services.ranking.rankers.ordering import rank_by_cosine, rank_by_score, stable_order
from services.ranking.rankers.score_fusion import (
    DEFAULT_RRF_K,
    METHODS,
    combmnz_fusion,
    combsum_fusion,
    fuse_results,
    normalize_scores,
    reciprocal_rank_fusion,
    weighted_fusion,
)

__all__ = [
    "DEFAULT_RRF_K",
    "METHODS",
    "combmnz_fusion",
    "combsum_fusion",
    "fuse_results",
    "normalize_scores",
    "rank_by_cosine",
    "rank_by_score",
    "reciprocal_rank_fusion",
    "stable_order",
    "weighted_fusion",
]
