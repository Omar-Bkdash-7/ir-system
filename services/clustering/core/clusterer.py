"""HDBSCAN clustering and the 2D UMAP projection.

The libraries are optional at import time (the base/retrieval images do not ship them): a missing library must turn
into a clear 503 on POST /cluster while GET /clusters and /status keep serving stored runs.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from services.clustering.config import HDBSCAN_METRICS
from services.clustering.errors import ClusteringUnavailableError, InvalidClusteringParameterError

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whichever image is running
    import hdbscan

    HDBSCAN_AVAILABLE = True
    HDBSCAN_IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover
    hdbscan = None  # type: ignore[assignment]
    HDBSCAN_AVAILABLE = False
    HDBSCAN_IMPORT_ERROR = str(exc)

try:  # pragma: no cover
    from umap import UMAP

    UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    UMAP = None  # type: ignore[assignment]
    UMAP_AVAILABLE = False

CLUSTERING_AVAILABLE = HDBSCAN_AVAILABLE and UMAP_AVAILABLE

MIN_CLUSTER_SIZE_FLOOR = 2
MIN_SAMPLES_FLOOR = 1


def require_clustering_available() -> None:
    if CLUSTERING_AVAILABLE:
        return
    missing = []
    if not HDBSCAN_AVAILABLE:
        missing.append(f"hdbscan ({HDBSCAN_IMPORT_ERROR or 'not installed'})")
    if not UMAP_AVAILABLE:
        missing.append("umap-learn (not installed)")
    raise ClusteringUnavailableError(
        "Clustering libraries are not installed in this image: "
        + ", ".join(missing)
        + ". Rebuild it with requirements/06-clustering.txt."
    )


def validate_parameters(
    *,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    n_keywords: int,
    n_docs: Optional[int] = None,
) -> None:
    """Reject what HDBSCAN would only fail on later with an opaque message."""
    if int(min_cluster_size) < MIN_CLUSTER_SIZE_FLOOR:
        raise InvalidClusteringParameterError(f"min_cluster_size must be >= {MIN_CLUSTER_SIZE_FLOOR}")
    if int(min_samples) < MIN_SAMPLES_FLOOR:
        raise InvalidClusteringParameterError(f"min_samples must be >= {MIN_SAMPLES_FLOOR}")
    if min_samples > min_cluster_size:
        raise InvalidClusteringParameterError("min_samples must not exceed min_cluster_size")
    if int(n_keywords) < 1:
        raise InvalidClusteringParameterError("n_keywords must be >= 1")
    if metric not in HDBSCAN_METRICS:
        raise InvalidClusteringParameterError(f"metric must be one of {list(HDBSCAN_METRICS)}, got {metric!r}")
    if n_docs is not None and int(min_cluster_size) > int(n_docs):
        raise InvalidClusteringParameterError(
            f"min_cluster_size ({min_cluster_size}) is larger than the dataset ({n_docs} documents)"
        )


def run_hdbscan(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster labels for every document; -1 means noise."""
    require_clustering_available()
    logger.info(
        "Running HDBSCAN on %d documents: min_cluster_size=%d min_samples=%d metric=%s",
        len(embeddings),
        min_cluster_size,
        min_samples,
        metric,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric=metric,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        # No prediction_data: approximate_predict is never used, and it keeps the tree and exemplars in memory.
    )
    labels = np.asarray(clusterer.fit_predict(embeddings), dtype=np.int64)
    if labels.shape[0] != embeddings.shape[0]:
        raise InvalidClusteringParameterError(
            f"HDBSCAN returned {labels.shape[0]} labels for {embeddings.shape[0]} documents"
        )
    n_clusters = len(set(labels.tolist()) - {-1})
    logger.info("HDBSCAN finished: %d clusters, %d noise documents", n_clusters, int(np.sum(labels == -1)))
    return labels


def _svd_projection(embeddings: np.ndarray) -> np.ndarray:
    """Deterministic 2D fallback (principal components) for tiny or degenerate datasets."""
    centred = embeddings.astype(np.float64) - embeddings.mean(axis=0, keepdims=True)
    if centred.shape[1] == 1:
        return np.column_stack([centred[:, 0], np.zeros(len(centred))]).astype(np.float32)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    coordinates = centred @ components[:2].T
    if coordinates.shape[1] < 2:  # a single document (or a single dimension): pad the missing axis with zeros
        coordinates = np.column_stack([coordinates, np.zeros(len(coordinates))])
    return np.ascontiguousarray(coordinates, dtype=np.float32)


def project_2d(
    embeddings: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """2D coordinates for the scatter plot. Falls back to PCA if UMAP refuses the data (too few documents,
    duplicate points), because a failed projection must not lose the clustering result."""
    n_docs = int(embeddings.shape[0])
    if not UMAP_AVAILABLE or n_docs < 4:
        logger.info("Using the PCA projection (documents=%d, umap available=%s)", n_docs, UMAP_AVAILABLE)
        return _svd_projection(embeddings)

    effective_neighbors = max(2, min(int(n_neighbors), n_docs - 1))
    try:
        model = UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=float(min_dist),
            metric=metric,
            random_state=random_state,
        )
        return np.asarray(model.fit_transform(embeddings), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - UMAP raises several unrelated types for degenerate inputs
        logger.warning("UMAP projection failed (%s); using the PCA projection instead", exc)
        return _svd_projection(embeddings)
