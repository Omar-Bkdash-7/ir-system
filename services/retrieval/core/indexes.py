"""The three index types behind one small interface.

    prepare(query)          resolve a request-level query (QueryContext) into this model's own form
    search(plan, k, opts)   top-k over the whole corpus            -> Hits(rows, scores)
    score(plan, rows, opts) scores of given rows (hybrid serial)   -> float64 array, 0.0 for rows == -1

Documents are addressed by ROW (their position in doc_ids); ids are only translated at the edges. Indexes are immutable
once built, so any number of request threads can search the same instance concurrently.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from common.config import get_dataset_specific_path
from common.exceptions import IndexNotFoundError, RetrievalError

from . import kernels
from .compact import CompactCache, Payload, build_bm25_payload, build_tfidf_payload, source_stamp, stamp_key
from .names import LABELS, dataset_slug

logger = logging.getLogger(__name__)

_NO_COLUMNS = np.empty(0, dtype=np.int64)
_NO_WEIGHTS = np.empty(0, dtype=np.float64)


class Hits(NamedTuple):
    rows: np.ndarray    # int64 row numbers, best first
    scores: np.ndarray  # float64, same length


EMPTY_HITS = Hits(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))


@dataclass(frozen=True)
class SearchOptions:
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass(frozen=True)
class TermPlan:
    """Query terms resolved against one lexical index: term ids (BM25) or vocabulary columns (TF-IDF) and weights."""

    ids: np.ndarray
    weights: np.ndarray

    @property
    def empty(self) -> bool:
        return self.ids.size == 0


def _touch(array: np.ndarray) -> None:
    """Read one value per memory page so a memory-mapped array is resident before the first query needs it."""
    flat = array.reshape(-1)
    if flat.size:
        flat[:: max(1, 4096 // max(1, flat.itemsize))].sum()


class BaseIndex:
    kind = ""

    def __init__(self, dataset: str, doc_ids: List[Any]) -> None:
        self.dataset = dataset
        self.doc_ids = doc_ids
        self._row_map: Optional[Dict[Any, int]] = None
        self._row_lock = threading.Lock()

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def nbytes(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def ids_for(self, rows: np.ndarray) -> List[Any]:
        ids = self.doc_ids
        return [ids[row] for row in rows.tolist()]

    def rows_for(self, doc_ids: Sequence[Any]) -> np.ndarray:
        """Row of each id in this index, -1 for ids it does not contain. The id -> row map is built once, on first use."""
        mapping = self._row_map
        if mapping is None:
            with self._row_lock:
                if self._row_map is None:
                    self._row_map = {doc_id: row for row, doc_id in enumerate(self.doc_ids)}
                mapping = self._row_map
        get = mapping.get
        return np.fromiter((get(doc_id, -1) for doc_id in doc_ids), dtype=np.int64, count=len(doc_ids))

    def _estimate_python_bytes(self, n_terms: int = 0) -> int:
        return 80 * len(self.doc_ids) + 100 * n_terms

    def warm(self) -> None:  # pragma: no cover - overridden
        pass

    def stats(self) -> Dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError


# --------------------------------------------------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------------------------------------------------
class Bm25Index(BaseIndex):
    kind = "bm25"

    def __init__(self, dataset: str, payload: Payload) -> None:
        super().__init__(dataset, payload.json["doc_ids"])
        arrays = payload.arrays
        terms = payload.json["terms"]
        self._terms: Dict[str, int] = {term: number for number, term in enumerate(terms)}
        self._offsets = arrays["offsets"]
        self._doc_idx = arrays["doc_idx"]
        self._tf = arrays["tf"]
        self._idf = arrays["idf"]
        self._doc_len = arrays["doc_len"]
        self._avg_len = float(payload.meta["avg_doc_length"])
        self._meta = payload.meta
        self._arrays_nbytes = payload.nbytes
        self._python_nbytes = self._estimate_python_bytes(len(terms))

    @property
    def nbytes(self) -> int:
        return self._arrays_nbytes + self._python_nbytes

    def prepare(self, query: Any) -> TermPlan:
        counts = Counter(query.lexical_tokens(self.kind))
        ids: List[int] = []
        weights: List[float] = []
        for term, count in counts.items():
            number = self._terms.get(term)
            if number is not None:  # words that are not in the index cannot match anything
                ids.append(number)
                weights.append(float(count))
        if not ids:
            return TermPlan(_NO_COLUMNS, _NO_WEIGHTS)
        return TermPlan(np.asarray(ids, dtype=np.int64), np.asarray(weights, dtype=np.float64))

    def _scores(self, plan: TermPlan, opts: SearchOptions) -> np.ndarray:
        return kernels.bm25_scores(
            self._offsets, self._doc_idx, self._tf, self._idf, self._doc_len, self._avg_len,
            plan.ids, plan.weights, opts.bm25_k1, opts.bm25_b,
        )

    def search(self, plan: TermPlan, k: int, opts: SearchOptions) -> Hits:
        if plan.empty:
            return EMPTY_HITS
        rows, scores = kernels.top_k(self._scores(plan, opts), k, positive_only=True)
        return Hits(rows, scores)

    def score(self, plan: TermPlan, rows: np.ndarray, opts: SearchOptions) -> np.ndarray:
        if plan.empty:
            return np.zeros(rows.shape[0], dtype=np.float64)
        return kernels.gather_scores(self._scores(plan, opts), rows)

    def warm(self) -> None:
        for array in (self._offsets, self._doc_idx, self._tf, self._doc_len):
            _touch(array)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_docs": self.n_docs,
            "vocab_size": len(self._terms),
            "avg_doc_length": round(self._avg_len, 2),
            "postings": int(self._meta.get("postings", self._doc_idx.shape[0])),
            "memory_mb": round(self.nbytes / (1024 * 1024), 2),
        }


# --------------------------------------------------------------------------------------------------------------------
# TF-IDF
# --------------------------------------------------------------------------------------------------------------------
class _NativeVectorizer:
    """Query -> (columns, unit-length weights) with numpy only, reproducing TfidfVectorizer.transform for a default
    vectorizer (word analyzer, unigrams, L2 norm). The index builder decides whether this is safe (meta.vectorizer.native)."""

    def __init__(self, terms: Dict[str, int], idf: np.ndarray, token_pattern: str, lowercase: bool, sublinear_tf: bool) -> None:
        self._terms = terms
        self._idf = idf
        self._findall = re.compile(token_pattern).findall
        self._lowercase = lowercase
        self._sublinear = sublinear_tf

    def __call__(self, tokens: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        text = " ".join(tokens)
        if self._lowercase:
            text = text.lower()
        counts = Counter(self._findall(text))
        columns: List[int] = []
        frequencies: List[float] = []
        for term, count in counts.items():
            column = self._terms.get(term)
            if column is not None:
                columns.append(column)
                frequencies.append(float(count))
        if not columns:
            return _NO_COLUMNS, _NO_WEIGHTS
        cols = np.asarray(columns, dtype=np.int64)
        weights = np.asarray(frequencies, dtype=np.float64)
        if self._sublinear:
            weights = 1.0 + np.log(weights)
        weights = weights * self._idf[cols]
        norm = float(np.sqrt(np.dot(weights, weights)))
        if norm == 0.0:
            return _NO_COLUMNS, _NO_WEIGHTS
        return cols, weights / norm


class _SklearnVectorizer:
    """Fallback for unusual vectorizer settings: ask the pickled vectorizer itself."""

    def __init__(self, vectorizer: Any) -> None:
        self._vectorizer = vectorizer

    def __call__(self, tokens: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        row = self._vectorizer.transform([" ".join(tokens)]).tocsr()
        if row.nnz == 0:
            return _NO_COLUMNS, _NO_WEIGHTS
        data = row.data.astype(np.float64)
        norm = float(np.sqrt(np.dot(data, data)))
        if norm == 0.0:
            return _NO_COLUMNS, _NO_WEIGHTS
        return row.indices.astype(np.int64), data / norm


class TfidfIndex(BaseIndex):
    kind = "tfidf"

    def __init__(self, dataset: str, payload: Payload, vectorizer_path: Path) -> None:
        super().__init__(dataset, payload.json["doc_ids"])
        arrays = payload.arrays
        self._offsets = arrays["offsets"]
        self._doc_idx = arrays["doc_idx"]
        self._values = arrays["values"]
        self._meta = payload.meta
        terms = payload.json["terms"]
        info = payload.meta["vectorizer"]
        if info.get("native") and "idf" in arrays:
            vocabulary = {term: column for column, term in enumerate(terms)}
            self._vectorize: Callable[[Sequence[str]], Tuple[np.ndarray, np.ndarray]] = _NativeVectorizer(
                vocabulary, arrays["idf"], info["token_pattern"], info["lowercase"], info["sublinear_tf"]
            )
            self._vocab_size = len(vocabulary)
            python_bytes = self._estimate_python_bytes(len(vocabulary))
        else:
            logger.info("TF-IDF index of %s uses the pickled vectorizer for query vectors (non-default settings)", dataset)
            with open(vectorizer_path, "rb") as handle:
                self._vectorize = _SklearnVectorizer(pickle.load(handle))
            self._vocab_size = len(terms)
            python_bytes = self._estimate_python_bytes(len(terms)) * 2
        self._nbytes = payload.nbytes + python_bytes

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def prepare(self, query: Any) -> TermPlan:
        columns, weights = self._vectorize(query.lexical_tokens(self.kind))
        return TermPlan(columns, weights)

    def _scores(self, plan: TermPlan) -> np.ndarray:
        return kernels.sparse_dot_scores(self._offsets, self._doc_idx, self._values, plan.ids, plan.weights, self.n_docs)

    def search(self, plan: TermPlan, k: int, opts: SearchOptions) -> Hits:
        if plan.empty:
            return EMPTY_HITS
        rows, scores = kernels.top_k(self._scores(plan), k, positive_only=True)
        return Hits(rows, scores)

    def score(self, plan: TermPlan, rows: np.ndarray, opts: SearchOptions) -> np.ndarray:
        if plan.empty:
            return np.zeros(rows.shape[0], dtype=np.float64)
        return kernels.gather_scores(self._scores(plan), rows)

    def warm(self) -> None:
        for array in (self._offsets, self._doc_idx, self._values):
            _touch(array)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_docs": self.n_docs,
            "vocab_size": self._vocab_size,
            "postings": int(self._meta.get("nnz", self._doc_idx.shape[0])),
            "memory_mb": round(self.nbytes / (1024 * 1024), 2),
        }


# --------------------------------------------------------------------------------------------------------------------
# Dense embeddings
# --------------------------------------------------------------------------------------------------------------------
class DenseIndex(BaseIndex):
    kind = "embedding"

    def __init__(self, dataset: str, matrix: np.ndarray, doc_ids: List[Any]) -> None:
        super().__init__(dataset, doc_ids)
        self._matrix = matrix
        self._python_nbytes = self._estimate_python_bytes()

    @property
    def dim(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self._matrix.nbytes) + self._python_nbytes

    def prepare(self, query: Any) -> np.ndarray:
        vector = query.dense_vector()
        if vector.shape[0] != self.dim:
            raise RetrievalError(
                f"Model dimension mismatch: model={vector.shape[0]}, index={self.dim}. "
                "Rebuild the index with the current model."
            )
        return vector

    def search(self, plan: np.ndarray, k: int, opts: SearchOptions) -> Hits:
        scores = kernels.dense_scores(self._matrix, plan)
        if not np.isfinite(scores).all():
            raise RetrievalError("Invalid similarity scores computed")
        rows, values = kernels.top_k(scores, k)
        return Hits(rows, values)

    def score(self, plan: np.ndarray, rows: np.ndarray, opts: SearchOptions) -> np.ndarray:
        out = np.zeros(rows.shape[0], dtype=np.float64)
        valid = rows >= 0
        if valid.any():
            out[valid] = self._matrix[rows[valid]] @ plan
        return out

    def warm(self) -> None:
        _touch(self._matrix)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_docs": self.n_docs,
            "embedding_dimension": self.dim,
            "memory_mb": round(int(self._matrix.nbytes) / (1024 * 1024), 2),
        }


# --------------------------------------------------------------------------------------------------------------------
# locating and loading
# --------------------------------------------------------------------------------------------------------------------
_SOURCES = {
    "bm25": (("indices.bm25_index", "data/indices/{dataset}/bm25/bm25_index.pkl"),),
    "tfidf": (
        ("indices.tfidf_vectorizer", "data/indices/{dataset}/tfidf/tfidf_vectorizer.pkl"),
        ("indices.tfidf_matrix", "data/indices/{dataset}/tfidf/tfidf_matrix.npz"),
        ("indices.tfidf_doc_ids", "data/indices/{dataset}/tfidf/tfidf_doc_ids.json"),
    ),
    "embedding": (
        ("indices.embeddings", "data/indices/{dataset}/embedding/embeddings.npy"),
        ("indices.doc_id_map", "data/indices/{dataset}/embedding/doc_id_map.json"),
    ),
}


def source_files(kind: str, dataset: str) -> List[Path]:
    """Files the index of `kind` is built from, located through the same config keys the service always used."""
    return [Path(get_dataset_specific_path(key, dataset, default)) for key, default in _SOURCES[kind]]


def load_index(kind: str, dataset: str, *, embedding_mmap: bool, cache: CompactCache) -> BaseIndex:
    """Load an index, building (and publishing) its compact arrays on first use."""
    files = source_files(kind, dataset)
    for path in files:
        if not path.exists():
            raise IndexNotFoundError(f"{LABELS[kind]} index not found at {path}")
    if kind == "embedding":
        return _load_dense(dataset, files, embedding_mmap)

    stamp = source_stamp(files)
    key = stamp_key(stamp)
    slug = dataset_slug(dataset)
    payload = cache.load(slug, kind, key)
    if payload is None:
        payload = build_bm25_payload(files[0]) if kind == "bm25" else build_tfidf_payload(files[0], files[1], files[2])
        if cache.save(slug, kind, key, payload, stamp) is not None:
            payload = cache.load(slug, kind, key) or payload  # re-open memory-mapped and let the in-RAM copy go
    if kind == "bm25":
        return Bm25Index(dataset, payload)
    return TfidfIndex(dataset, payload, files[0])


def _load_dense(dataset: str, files: List[Path], mmap: bool) -> DenseIndex:
    embeddings_path, id_map_path = files
    with open(id_map_path, "r", encoding="utf-8") as handle:
        doc_ids = json.load(handle)
    matrix = np.load(embeddings_path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if matrix.ndim != 2:
        raise RetrievalError(f"Embedding matrix {embeddings_path.name} must be 2-dimensional, got shape {matrix.shape}")
    if matrix.shape[0] != len(doc_ids):
        raise RetrievalError(
            f"Embedding index is inconsistent: {matrix.shape[0]} embeddings but {len(doc_ids)} document ids "
            "(an index rebuild may be in progress)"
        )
    if matrix.dtype != np.float32:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    return DenseIndex(dataset, matrix, doc_ids)
