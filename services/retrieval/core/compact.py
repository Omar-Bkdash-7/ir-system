"""Compact, memory-mappable copy of the BM25 and TF-IDF indices.

The pickles / npz files written by the indexing service stay the single source of truth. On first use they are converted
ONCE into flat numpy arrays (CSR-style postings) kept under ``<cache_dir>/<dataset-slug>/<kind>-<key>/``. ``key`` hashes
the name, mtime and size of every source file, so a rebuilt index gets a new key and a stale cache can never be read.

The cache is derived data: it can be deleted at any time. Publishing is atomic (temporary directory + rename), so
concurrent builders and crashes are harmless. When the cache directory is not writable the arrays simply stay in memory
(slower cold start, identical results).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from common.exceptions import RetrievalError

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
MANIFEST_NAME = "MANIFEST.json"
_STALE_TMP_SECONDS = 3600


@dataclass
class Payload:
    """What an index needs: numpy arrays, JSON-able lists and a few scalars."""

    arrays: Dict[str, np.ndarray]
    json: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return int(sum(array.nbytes for array in self.arrays.values()))


# --------------------------------------------------------------------------------------------------------------------
# cache keys and storage
# --------------------------------------------------------------------------------------------------------------------
def source_stamp(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """Identity of the source files: name, modification time (ns) and size. Raises FileNotFoundError if one is missing."""
    stamp = []
    for path in paths:
        info = Path(path).stat()
        stamp.append({"name": Path(path).name, "mtime_ns": info.st_mtime_ns, "size": info.st_size})
    return stamp


def stamp_key(stamp: List[Dict[str, Any]]) -> str:
    blob = json.dumps({"format": FORMAT_VERSION, "files": stamp}, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


class CompactCache:
    """Directory of published payloads. ``root=None`` disables persistence (everything stays in memory)."""

    def __init__(self, root: Optional[Path]) -> None:
        self.root = Path(root) if root else None
        self._warned = False

    def _directory(self, slug: str, kind: str, key: str) -> Path:
        assert self.root is not None
        return self.root / slug / f"{kind}-{key}"

    def load(self, slug: str, kind: str, key: str, *, mmap: bool = True) -> Optional[Payload]:
        """The published payload for exactly this source version, or None (missing, partial or unreadable)."""
        if self.root is None:
            return None
        directory = self._directory(slug, kind, key)
        try:
            info = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
            if info.get("format") != FORMAT_VERSION:
                return None
            arrays = {name: self._load_array(directory / f"{name}.npy", mmap) for name in info["arrays"]}
            blobs = {name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8")) for name in info["json"]}
        except (OSError, ValueError, KeyError):
            return None
        return Payload(arrays, blobs, info.get("meta", {}))

    @staticmethod
    def _load_array(path: Path, mmap: bool) -> np.ndarray:
        if mmap:
            try:
                return np.load(path, mmap_mode="r", allow_pickle=False)
            except ValueError:  # e.g. zero-length arrays cannot be memory-mapped on every platform
                pass
        return np.load(path, allow_pickle=False)

    def save(self, slug: str, kind: str, key: str, payload: Payload, stamp: List[Dict[str, Any]]) -> Optional[Path]:
        """Publish atomically. Returns the directory, or None when the cache is disabled or not writable."""
        if self.root is None:
            return None
        directory = self._directory(slug, kind, key)
        tmp = directory.with_name(f"{directory.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        try:
            tmp.mkdir(parents=True, exist_ok=False)
            for name, array in payload.arrays.items():
                np.save(tmp / f"{name}.npy", np.ascontiguousarray(array), allow_pickle=False)
            for name, blob in payload.json.items():
                (tmp / f"{name}.json").write_text(json.dumps(blob, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            manifest = {
                "format": FORMAT_VERSION,
                "kind": kind,
                "key": key,
                "created": time.time(),
                "stamp": stamp,
                "arrays": {name: {"dtype": str(a.dtype), "shape": list(a.shape)} for name, a in payload.arrays.items()},
                "json": list(payload.json),
                "meta": payload.meta,
            }
            (tmp / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")  # written last: marks "complete"
            try:
                os.replace(tmp, directory)
            except OSError:
                if not (directory / MANIFEST_NAME).exists():
                    raise
                shutil.rmtree(tmp, ignore_errors=True)  # somebody else published the same version first: keep theirs
        except OSError as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            if not self._warned:
                self._warned = True
                logger.warning("Compact cache %s is not writable (%s): index arrays stay in memory only", self.root, exc)
            return None
        self._prune(slug, kind, directory.name)
        return directory

    def _prune(self, slug: str, kind: str, keep: str) -> None:
        """Remove older versions of the same index and abandoned temporary directories."""
        try:
            for entry in (self.root / slug).iterdir():  # type: ignore[operator]
                if not entry.name.startswith(f"{kind}-") or entry.name == keep:
                    continue
                if ".tmp-" in entry.name and time.time() - entry.stat().st_mtime < _STALE_TMP_SECONDS:
                    continue  # probably another process is still writing it
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass

    def clear(self, slug: Optional[str] = None) -> int:
        """Delete the cache (of one dataset or all). Returns the number of directories removed."""
        if self.root is None or not self.root.exists():
            return 0
        targets = [self.root / slug] if slug else [entry for entry in self.root.iterdir() if entry.is_dir()]
        removed = 0
        for target in targets:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                removed += 1
        return removed


# --------------------------------------------------------------------------------------------------------------------
# BM25: {term: [(doc_id, tf)]} pickle  ->  CSR-style arrays
# --------------------------------------------------------------------------------------------------------------------
def _smallest_unsigned(maximum: int) -> Any:
    for dtype in (np.uint8, np.uint16, np.uint32):
        if maximum <= np.iinfo(dtype).max:
            return dtype
    return np.uint64


def build_bm25_payload(pickle_path: Path) -> Payload:
    started = time.perf_counter()
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)
    try:
        inverted = data["inverted_index"]
        raw_ids = list(data["doc_ids"])
        doc_lengths = data["doc_lengths"]
        avg_len = float(data["avg_doc_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RetrievalError(f"BM25 index {pickle_path} is missing required data: {exc!r}") from exc
    idf_scores = data.get("idf_scores") or {}
    doc_freq = data.get("doc_frequencies") or {}
    total_docs = int(data.get("total_docs") or len(raw_ids))
    if avg_len <= 0:
        logger.error("Invalid avg_doc_length (%s) in %s; using 1.0", avg_len, pickle_path)
        avg_len = 1.0

    row_of: Dict[Any, int] = {}
    doc_ids: List[Any] = []
    for doc_id in raw_ids:
        if doc_id not in row_of:
            row_of[doc_id] = len(doc_ids)
            doc_ids.append(doc_id)
    if len(doc_ids) != len(raw_ids):
        logger.warning("BM25 index %s lists %d duplicate document ids; scores are merged per id", pickle_path, len(raw_ids) - len(doc_ids))

    terms = list(inverted.keys())
    n_postings = sum(len(entries) for entries in inverted.values())
    offsets = np.zeros(len(terms) + 1, dtype=np.int64)
    doc_idx = np.empty(n_postings, dtype=np.int32)
    tf_raw = np.empty(n_postings, dtype=np.int64)
    idf = np.empty(len(terms), dtype=np.float64)
    position = 0
    for number, term in enumerate(terms):
        entries = inverted[term]
        count = len(entries)
        if count:
            rows = []
            for doc_id, _tf in entries:
                row = row_of.get(doc_id)
                if row is None:  # posting for an id missing from doc_ids: the old code still scored it, so keep it
                    row = len(doc_ids)
                    row_of[doc_id] = row
                    doc_ids.append(doc_id)
                rows.append(row)
            doc_idx[position:position + count] = rows
            tf_raw[position:position + count] = [entry[1] for entry in entries]
        position += count
        offsets[number + 1] = position
        if term in idf_scores:  # same precedence as the former BM25Retriever.compute_idf
            idf[number] = idf_scores[term]
        elif term in doc_freq:
            df = doc_freq[term]
            idf[number] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
        else:
            idf[number] = 0.0

    doc_len = np.fromiter((doc_lengths.get(doc_id, avg_len) for doc_id in doc_ids), dtype=np.float64, count=len(doc_ids))
    tf = tf_raw.astype(_smallest_unsigned(int(tf_raw.max()) if n_postings else 0))
    logger.info("Built compact BM25 arrays from %s: %d docs, %d terms, %d postings in %.1fs",
                pickle_path.name, len(doc_ids), len(terms), n_postings, time.perf_counter() - started)
    return Payload(
        arrays={"offsets": offsets, "doc_idx": doc_idx, "tf": tf, "idf": idf, "doc_len": doc_len},
        json={"terms": terms, "doc_ids": doc_ids},
        meta={"n_docs": len(doc_ids), "vocab_size": len(terms), "postings": n_postings, "avg_doc_length": avg_len},
    )


# --------------------------------------------------------------------------------------------------------------------
# TF-IDF: sklearn vectorizer + L2-normalised CSR matrix  ->  term-major postings + native query vectoriser data
# --------------------------------------------------------------------------------------------------------------------
def _extract_idf(vectorizer: Any) -> Optional[np.ndarray]:
    try:
        return np.asarray(vectorizer.idf_, dtype=np.float64).ravel()
    except AttributeError:
        # a pickle written by an older scikit-learn keeps the weights in a diagonal matrix
        diagonal = getattr(getattr(vectorizer, "_tfidf", None), "_idf_diag", None)
        if diagonal is not None:
            return np.asarray(diagonal.sum(axis=0), dtype=np.float64).ravel()
    return None


def _vectorizer_meta(vectorizer: Any, has_idf: bool) -> Dict[str, Any]:
    def get(name: str, default: Any = None) -> Any:
        return getattr(vectorizer, name, default)

    token_pattern = get("token_pattern")
    native = (
        has_idf
        and get("analyzer", "word") == "word"
        and tuple(get("ngram_range", (1, 1))) == (1, 1)
        and not get("stop_words")
        and get("preprocessor") is None
        and get("tokenizer") is None
        and not get("strip_accents")
        and not get("binary", False)
        and bool(get("use_idf", True))
        and get("norm") == "l2"
        and isinstance(token_pattern, str)
    )
    return {
        "native": bool(native),
        "token_pattern": token_pattern if isinstance(token_pattern, str) else None,
        "lowercase": bool(get("lowercase", True)),
        "sublinear_tf": bool(get("sublinear_tf", False)),
    }


def _unit_rows(matrix: Any) -> Any:
    """Make every non-empty row unit length (a no-op for matrices written by the indexer, which are normalised)."""
    import scipy.sparse as sp

    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1), dtype=np.float64).ravel())
    nonzero = norms > 0
    if nonzero.any() and float(np.max(np.abs(norms[nonzero] - 1.0))) > 1e-6:
        scale = np.ones_like(norms)
        scale[nonzero] = 1.0 / norms[nonzero]
        matrix = (sp.diags(scale) @ matrix).tocsr()
    return matrix


def build_tfidf_payload(vectorizer_path: Path, matrix_path: Path, doc_ids_path: Path) -> Payload:
    import scipy.sparse as sp

    started = time.perf_counter()
    with open(doc_ids_path, "r", encoding="utf-8") as handle:
        doc_ids = json.load(handle)
    matrix = sp.csr_matrix(sp.load_npz(matrix_path))
    if matrix.shape[0] != len(doc_ids):
        raise RetrievalError(
            f"TF-IDF index is inconsistent: matrix has {matrix.shape[0]} rows but {doc_ids_path.name} lists {len(doc_ids)} "
            "documents (an index rebuild may be in progress)"
        )
    with open(vectorizer_path, "rb") as handle:
        vectorizer = pickle.load(handle)
    vocabulary = vectorizer.vocabulary_
    n_columns = matrix.shape[1]
    terms: List[Optional[str]] = [None] * n_columns
    for term, column in vocabulary.items():
        if int(column) >= n_columns:
            raise RetrievalError(f"TF-IDF index is inconsistent: vectorizer and matrix come from different builds ({vectorizer_path.name})")
        terms[int(column)] = term
    if any(term is None for term in terms):
        raise RetrievalError(f"TF-IDF index is inconsistent: vectorizer and matrix come from different builds ({vectorizer_path.name})")

    idf = _extract_idf(vectorizer)
    if idf is not None and idf.shape[0] != n_columns:
        idf = None
    vectorizer_meta = _vectorizer_meta(vectorizer, has_idf=idf is not None)

    by_term = _unit_rows(matrix).tocsc()
    by_term.sort_indices()
    arrays = {
        "offsets": by_term.indptr.astype(np.int64),
        "doc_idx": by_term.indices.astype(np.int32),
        "values": np.ascontiguousarray(by_term.data),
    }
    if idf is not None:
        arrays["idf"] = idf
    logger.info("Built compact TF-IDF arrays from %s: %d docs, %d terms, %d non-zeros in %.1fs",
                matrix_path.name, matrix.shape[0], n_columns, by_term.nnz, time.perf_counter() - started)
    return Payload(
        arrays=arrays,
        json={"terms": terms, "doc_ids": doc_ids},
        meta={"n_docs": int(matrix.shape[0]), "vocab_size": n_columns, "nnz": int(by_term.nnz), "vectorizer": vectorizer_meta},
    )
