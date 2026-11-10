"""Reading and validating the embedding index built by the indexing service.

HDBSCAN needs the full matrix, so the file is read (not memory-mapped) after a cheap sanity check: a truncated or
mismatched index must fail here with a 404/500 that names the real problem instead of crashing inside sklearn.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from services.clustering.core.paths import embedding_files
from services.clustering.errors import EmbeddingIndexMissingError, InvalidClusteringParameterError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingIndex:
    """A loaded, validated embedding index."""

    embeddings: np.ndarray
    doc_ids: List[str]
    fingerprint: str

    @property
    def n_docs(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])


def _file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def fingerprint(embeddings_path: Path, doc_id_map_path: Path) -> str:
    """Identity of an embedding index on disk.

    Size + mtime of both files change on every rebuild, which is all the cache needs to notice that clusters built
    from the previous index must not be served any more. The digest keeps the manifest short.
    """
    signature = f"{_file_signature(embeddings_path)}|{_file_signature(doc_id_map_path)}"
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]


def missing_embedding_files(indices_dir: Path, dataset: str) -> List[str]:
    """Paths of the index files that do not exist (empty list = the index was built)."""
    embeddings_path, doc_id_map_path = embedding_files(indices_dir, dataset)
    return [str(path) for path in (embeddings_path, doc_id_map_path) if not path.exists()]


def index_fingerprint(indices_dir: Path, dataset: str) -> str:
    """Fingerprint without reading the matrix: a cache lookup must not load a large .npy file.

    Raises EmbeddingIndexMissingError when the index was never built, so callers can tell
    "no index" (404 for POST /cluster) from "index changed" (stale cache).
    """
    embeddings_path, doc_id_map_path = embedding_files(indices_dir, dataset)
    missing = missing_embedding_files(indices_dir, dataset)
    if missing:
        raise EmbeddingIndexMissingError(
            f"Embedding index not found for dataset {dataset!r} ({', '.join(missing)}). "
            "Build embeddings first via the indexing service."
        )
    return fingerprint(embeddings_path, doc_id_map_path)


def load_embedding_index(indices_dir: Path, dataset: str) -> EmbeddingIndex:
    """Load <indices_dir>/<slug>/embedding and refuse anything HDBSCAN could not cluster."""
    embeddings_path, doc_id_map_path = embedding_files(indices_dir, dataset)
    missing = missing_embedding_files(indices_dir, dataset)
    if missing:
        raise EmbeddingIndexMissingError(
            f"Embedding index not found for dataset {dataset!r} ({', '.join(missing)}). "
            "Build embeddings first via the indexing service."
        )

    try:
        embeddings = np.load(embeddings_path, mmap_mode=None)
    except (OSError, ValueError) as exc:
        raise InvalidClusteringParameterError(f"Unreadable embedding file {embeddings_path}: {exc}") from exc

    with open(doc_id_map_path, "r", encoding="utf-8") as handle:
        try:
            doc_ids = json.load(handle)
        except json.JSONDecodeError as exc:
            raise InvalidClusteringParameterError(f"Unreadable doc id map {doc_id_map_path}: {exc}") from exc

    if not isinstance(doc_ids, list):
        raise InvalidClusteringParameterError(f"Doc id map {doc_id_map_path} is not a list of document ids")
    if embeddings.ndim != 2:
        raise InvalidClusteringParameterError(
            f"Embedding matrix must be 2-dimensional, got shape {embeddings.shape}"
        )
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise InvalidClusteringParameterError(f"Embedding matrix {embeddings_path} is empty (shape {embeddings.shape})")
    if embeddings.shape[0] != len(doc_ids):
        raise InvalidClusteringParameterError(
            f"Embedding index is inconsistent: {embeddings.shape[0]} vectors but {len(doc_ids)} document ids"
        )
    if not np.all(np.isfinite(embeddings)):
        raise InvalidClusteringParameterError(f"Embedding matrix {embeddings_path} contains NaN or infinite values")

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    logger.info(
        "Loaded embedding index for %s: %d documents, dimension %d", dataset, embeddings.shape[0], embeddings.shape[1]
    )
    return EmbeddingIndex(
        embeddings=embeddings,
        doc_ids=[str(doc_id) for doc_id in doc_ids],
        fingerprint=fingerprint(embeddings_path, doc_id_map_path),
    )
