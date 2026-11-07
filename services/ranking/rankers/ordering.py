"""Ordering by score: the primitive behind POST /rank (and the former `rank_by_cosine`).

Despite its old name nothing here computes a cosine: it orders results that already carry a score. Equal scores keep
their original relative order (the order a stable sort gives), so the output is deterministic.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import numpy as np


def stable_order(scores: np.ndarray) -> np.ndarray:
    """Positions that order `scores` best (highest) first; equal scores keep their original order."""
    return np.argsort(-scores, kind="stable")


def rank_by_score(results: Sequence[Mapping[str, Any]], top_k: Optional[int] = None) -> List[Mapping[str, Any]]:
    """`results` ordered by their "score", best first, cut to `top_k` (None = all). The items are returned as they are."""
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    if not results:
        return []
    scores = np.fromiter((item["score"] for item in results), dtype=np.float64, count=len(results))
    order = stable_order(scores)
    if top_k is not None:
        order = order[:top_k]
    return [results[position] for position in order.tolist()]


rank_by_cosine = rank_by_score  # the name the package used to export; same behaviour
