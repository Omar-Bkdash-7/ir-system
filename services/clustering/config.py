"""Runtime settings of the clustering service.

Every value has a default in code. It can be overridden in configs/config.yaml (section ``clustering:``) and, for
the deployment-level knobs, by an environment variable (the variable wins). No configuration change is required
to run the service.

    key (clustering.<key>)        environment variable               default
    ----------------------------  ---------------------------------  ------------------------
    data_dir                      CLUSTERING_DATA_DIR                data/clusters
    indices_dir                   CLUSTERING_INDICES_DIR             data/indices
    min_cluster_size              CLUSTERING_MIN_CLUSTER_SIZE        5
    min_samples                   CLUSTERING_MIN_SAMPLES             3
    metric                        CLUSTERING_METRIC                  euclidean
    n_keywords                    CLUSTERING_N_KEYWORDS              10
    representative_docs           CLUSTERING_REPRESENTATIVE_DOCS     3
    umap_n_neighbors              CLUSTERING_UMAP_N_NEIGHBORS        15
    umap_min_dist                 CLUSTERING_UMAP_MIN_DIST           0.1
    umap_metric                   CLUSTERING_UMAP_METRIC             cosine
    random_state                  CLUSTERING_RANDOM_STATE            42
    text_batch_size               CLUSTERING_TEXT_BATCH_SIZE         1000
    max_missing_text_ratio        CLUSTERING_MAX_MISSING_TEXT_RATIO  0.05
    db_pool_max                   CLUSTERING_DB_POOL_MAX             4
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Tuple, TypeVar

from common.config import get_config

logger = logging.getLogger(__name__)

#: HDBSCAN metrics this service accepts. The first entry is the default and the one a cosine-normalised
#: embedding index (the indexer normalises by default) behaves best with.
HDBSCAN_METRICS: Tuple[str, ...] = ("euclidean", "manhattan", "cosine")

T = TypeVar("T")


def _as_int(value: Any) -> int:
    return int(value)


def _as_positive_int(value: Any) -> int:
    number = int(value)
    if number < 1:
        raise ValueError("must be >= 1")
    return number


def _as_float(value: Any) -> float:
    return float(value)


def _as_metric(value: Any) -> str:
    metric = str(value).strip().lower()
    if metric not in HDBSCAN_METRICS:
        raise ValueError(f"must be one of {HDBSCAN_METRICS}")
    return metric


def _setting(env_name: str, key: str, default: T, cast: Callable[[Any], T]) -> T:
    """environment variable > configs/config.yaml (clustering.<key>) > default. A bad value is logged and ignored."""
    raw: Any = os.getenv(env_name)
    source = f"environment variable {env_name}"
    if raw is None or str(raw).strip() == "":
        raw = get_config(f"clustering.{key}", None)
        source = f"config key clustering.{key}"
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("Ignoring invalid value %r from %s (%s); using %r", raw, source, exc, default)
        return default


@dataclass(frozen=True)
class ClusteringSettings:
    data_dir: Path = Path("data/clusters")
    indices_dir: Path = Path("data/indices")
    min_cluster_size: int = 5
    min_samples: int = 3
    metric: str = "euclidean"
    n_keywords: int = 10
    representative_docs: int = 3
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    random_state: int = 42
    text_batch_size: int = 1000
    max_missing_text_ratio: float = 0.05
    db_pool_max: int = 4

    @classmethod
    def load(cls) -> "ClusteringSettings":
        return cls(
            data_dir=_setting("CLUSTERING_DATA_DIR", "data_dir", cls.data_dir, Path),
            indices_dir=_setting("CLUSTERING_INDICES_DIR", "indices_dir", cls.indices_dir, Path),
            min_cluster_size=_setting("CLUSTERING_MIN_CLUSTER_SIZE", "min_cluster_size", cls.min_cluster_size, _as_positive_int),
            min_samples=_setting("CLUSTERING_MIN_SAMPLES", "min_samples", cls.min_samples, _as_positive_int),
            metric=_setting("CLUSTERING_METRIC", "metric", cls.metric, _as_metric),
            n_keywords=_setting("CLUSTERING_N_KEYWORDS", "n_keywords", cls.n_keywords, _as_positive_int),
            representative_docs=_setting("CLUSTERING_REPRESENTATIVE_DOCS", "representative_docs", cls.representative_docs, _as_int),
            umap_n_neighbors=_setting("CLUSTERING_UMAP_N_NEIGHBORS", "umap_n_neighbors", cls.umap_n_neighbors, _as_positive_int),
            umap_min_dist=_setting("CLUSTERING_UMAP_MIN_DIST", "umap_min_dist", cls.umap_min_dist, _as_float),
            umap_metric=str(get_config("clustering.umap_metric", cls.umap_metric)),
            random_state=_setting("CLUSTERING_RANDOM_STATE", "random_state", cls.random_state, _as_int),
            text_batch_size=_setting("CLUSTERING_TEXT_BATCH_SIZE", "text_batch_size", cls.text_batch_size, _as_positive_int),
            max_missing_text_ratio=_setting(
                "CLUSTERING_MAX_MISSING_TEXT_RATIO", "max_missing_text_ratio", cls.max_missing_text_ratio, _as_float
            ),
            db_pool_max=_setting("CLUSTERING_DB_POOL_MAX", "db_pool_max", cls.db_pool_max, _as_positive_int),
        )


@lru_cache(maxsize=1)
def get_settings() -> ClusteringSettings:
    return ClusteringSettings.load()


def reset_settings_cache() -> None:
    """For tests: forget the cached settings so the next get_settings() re-reads the environment."""
    get_settings.cache_clear()
