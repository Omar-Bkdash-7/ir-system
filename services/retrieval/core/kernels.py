"""Vectorised scoring kernels: the hot loops of the retrieval service.

Pure functions of numpy arrays (no I/O, no framework, no shared state), so they are thread-safe and easy to test.
Ranking ties are broken by row index (corpus order), which makes every result list deterministic.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

_NO_ROWS = np.empty(0, dtype=np.int64)
_NO_SCORES = np.empty(0, dtype=np.float64)


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest `values`, ordered by (-value, index). `values` must not contain NaN."""
    n = values.shape[0]
    if k >= n:
        chosen = np.arange(n, dtype=np.int64)
    else:
        kth = np.partition(values, n - k)[n - k]  # value of the k-th largest element
        above = np.flatnonzero(values > kth)
        ties = np.flatnonzero(values == kth)[: k - above.size]
        chosen = np.concatenate((above, ties))
    return chosen[np.lexsort((chosen, -values[chosen]))]


def top_k(scores: np.ndarray, k: int, *, positive_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """(rows, scores) of the k best entries, best first.

    positive_only=True ignores entries that are not > 0: documents no query term touched (BM25, TF-IDF) are not hits.
    """
    if k <= 0 or scores.shape[0] == 0:
        return _NO_ROWS, _NO_SCORES
    if positive_only:
        candidates = np.flatnonzero(scores > 0)
        if candidates.size == 0:
            return _NO_ROWS, _NO_SCORES
        rows = candidates[_top_k_indices(scores[candidates], k)]
    else:
        rows = _top_k_indices(scores, k)
    return rows, scores[rows].astype(np.float64)


def _accumulate(docs: List[np.ndarray], contributions: List[np.ndarray], n_docs: int) -> np.ndarray:
    if not docs:
        return np.zeros(n_docs, dtype=np.float64)
    if len(docs) == 1:
        return np.bincount(docs[0], weights=contributions[0], minlength=n_docs)
    return np.bincount(np.concatenate(docs), weights=np.concatenate(contributions), minlength=n_docs)


def bm25_scores(
    offsets: np.ndarray,
    doc_idx: np.ndarray,
    tf: np.ndarray,
    idf: np.ndarray,
    doc_len: np.ndarray,
    avg_doc_len: float,
    term_ids: np.ndarray,
    term_weights: np.ndarray,
    k1: float,
    b: float,
) -> np.ndarray:
    """Okapi BM25 score of every document (0.0 for documents that contain no query term).

    Postings are stored CSR-style: the postings of term t are doc_idx[offsets[t]:offsets[t+1]] with term frequencies
    tf[...]. `term_weights` is the number of times the term occurs in the query (a repeated query word counts again,
    exactly like the loop this replaces). Only the postings of the query terms are touched.
    """
    docs: List[np.ndarray] = []
    contributions: List[np.ndarray] = []
    for term, weight in zip(term_ids.tolist(), term_weights.tolist()):
        start, end = int(offsets[term]), int(offsets[term + 1])
        if end <= start:
            continue
        d = doc_idx[start:end]
        f = tf[start:end].astype(np.float64)
        length_norm = k1 * (1.0 - b + b * (doc_len[d] / avg_doc_len))
        contributions.append((weight * float(idf[term])) * (f * (k1 + 1.0)) / (f + length_norm))
        docs.append(d)
    return _accumulate(docs, contributions, doc_len.shape[0])


def sparse_dot_scores(
    offsets: np.ndarray,
    doc_idx: np.ndarray,
    values: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
    n_docs: int,
) -> np.ndarray:
    """score[d] = sum over query terms t of weights[t] * M[d, t], reading only the postings of those terms.

    Used for TF-IDF cosine similarity: the stored matrix is stored transposed (one posting list per term) and its rows
    are unit length, so the dot product with the unit query vector IS the cosine similarity.
    """
    docs: List[np.ndarray] = []
    contributions: List[np.ndarray] = []
    for column, weight in zip(columns.tolist(), weights.tolist()):
        start, end = int(offsets[column]), int(offsets[column + 1])
        if end <= start:
            continue
        docs.append(doc_idx[start:end])
        contributions.append(values[start:end] * weight)
    return _accumulate(docs, contributions, n_docs)


def dense_scores(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Inner product of every row of `matrix` with `vector` (cosine similarity for unit-length rows)."""
    return np.asarray(matrix @ vector)


def gather_scores(scores: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """scores[rows] as float64, with 0.0 for rows < 0 (documents that are not part of this index)."""
    out = np.zeros(rows.shape[0], dtype=np.float64)
    valid = rows >= 0
    if valid.any():
        out[valid] = scores[rows[valid]]
    return out
