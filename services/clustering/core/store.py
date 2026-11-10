"""On-disk store of clustering runs.

Layout, one directory per dataset (see core/paths.py for the slug rule):

    data/clusters/<slug>/clusters.json              manifest: parameters, index fingerprint, labels, doc ids
    data/clusters/<slug>/run-<run_id>-embeddings_2d.npy
    data/clusters/<slug>/run-<run_id>-topics.json

The manifest is written last and names one run_id, and readers only open that run's derived files. A crash in the
middle of a run therefore leaves an unused file next to an intact older run (cleaned on the next save), never a
manifest that points at somebody else's topics. Every file is written to a temp name and renamed.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from services.clustering.core.paths import dataset_dir, run_artifact
from services.clustering.core.topics import TopicExtraction
from services.clustering.schemas import ClusterParams, TopicInfo

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
#: Bumped whenever the clustering/topic pipeline changes in a way that makes older runs wrong.
#: A run from another pipeline version is recomputed instead of being served from the cache.
PIPELINE_VERSION = 2
MANIFEST_NAME = "clusters.json"
#: Small file with the numbers /status and /datasets need, so polling them never parses the label arrays.
SUMMARY_NAME = "summary.json"
EMBEDDINGS_2D_KIND = "embeddings_2d.npy"
TOPICS_KIND = "topics.json"


@dataclass(frozen=True)
class ClusterSummary:
    """Everything /status and /datasets report, without touching the label arrays."""

    dataset: str
    timestamp: str
    params: ClusterParams
    index_fingerprint: str
    pipeline_version: int
    n_documents: int
    n_clusters: int
    n_noise: int
    topics_available: bool


@dataclass(frozen=True)
class ClusterRun:
    """One stored clustering run, with its derived data loaded alongside the manifest."""

    dataset: str
    slug: str
    run_id: str
    timestamp: str
    params: ClusterParams
    index_fingerprint: str
    pipeline_version: int
    doc_ids: List[str]
    cluster_labels: List[int]
    n_clusters: int
    n_noise: int
    topics_available: bool
    topics_method: Optional[str]
    warnings: List[str] = field(default_factory=list)
    embeddings_2d: Optional[np.ndarray] = None
    topics: Dict[int, TopicInfo] = field(default_factory=dict)

    @property
    def n_documents(self) -> int:
        return len(self.cluster_labels)

    def matches(self, params: ClusterParams, index_fingerprint: str, pipeline_version: int) -> bool:
        """True when serving this run is the correct answer for a request with these parameters."""
        return (
            self.params == params
            and self.index_fingerprint == index_fingerprint
            and self.pipeline_version == pipeline_version
        )

    def document_index(self) -> Dict[str, int]:
        """{doc_id: position}, so /document does not scan the list on every call."""
        return {doc_id: position for position, doc_id in enumerate(self.doc_ids)}


def _atomic_write(path: Path, writer) -> None:
    """Write through a temp file next to `path` and rename it: no reader ever sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "wb") as handle:
            writer(handle)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload) -> None:
    _atomic_write(path, lambda handle: handle.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")))


def save(
    data_dir: Path,
    dataset: str,
    *,
    run_id: str,
    params: ClusterParams,
    index_fingerprint: str,
    doc_ids: Sequence[str],
    cluster_labels: np.ndarray,
    extraction: TopicExtraction,
    embeddings_2d: Optional[np.ndarray] = None,
    warnings: Optional[Sequence[str]] = None,
) -> ClusterRun:
    """Persist one run and return it (labels already converted to plain ints)."""
    directory = dataset_dir(data_dir, dataset)
    directory.mkdir(parents=True, exist_ok=True)

    labels = [int(label) for label in np.asarray(cluster_labels).tolist()]
    doc_ids = [str(doc_id) for doc_id in doc_ids]
    if len(labels) != len(doc_ids):
        raise ValueError(f"{len(labels)} labels for {len(doc_ids)} document ids")

    warnings = list(warnings or [])
    embeddings_2d = None if embeddings_2d is None else np.asarray(embeddings_2d, dtype=np.float32)
    if embeddings_2d is not None and embeddings_2d.shape[0] != len(labels):
        raise ValueError(
            f"2D projection has {embeddings_2d.shape[0]} points but there are {len(labels)} documents"
        )

    # Derived data first (named after this run), manifest last: the manifest is the commit point.
    if embeddings_2d is not None:
        target = run_artifact(directory, run_id, EMBEDDINGS_2D_KIND)
        _atomic_write(target, lambda handle: np.save(handle, embeddings_2d))

    topics: Dict[int, TopicInfo] = dict(extraction.topics) if extraction.available else {}
    if topics:
        payload = {
            str(cluster_id): {
                "topic_id": topic.topic_id,
                "keywords": topic.keywords,
                "representative_docs": topic.representative_docs,
            }
            for cluster_id, topic in topics.items()
        }
        _atomic_write(run_artifact(directory, run_id, TOPICS_KIND), lambda handle: handle.write(
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        ))

    n_clusters = len(set(labels) - {-1})
    n_noise = sum(1 for label in labels if label == -1)
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "slug": dataset_dir(data_dir, dataset).name,
        "run_id": run_id,
        "timestamp": timestamp,
        "params": params.model_dump(),
        "index_fingerprint": index_fingerprint,
        "pipeline_version": PIPELINE_VERSION,
        "n_documents": len(labels),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "topics_available": extraction.available,
        "topics_method": extraction.method,
        "warnings": warnings,
        "doc_ids": doc_ids,
        "cluster_labels": labels,
    }
    _write_json(directory / MANIFEST_NAME, manifest)
    _write_json(
        directory / SUMMARY_NAME,
        {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "dataset": dataset,
            "timestamp": timestamp,
            "params": params.model_dump(),
            "index_fingerprint": index_fingerprint,
            "n_documents": len(labels),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "topics_available": extraction.available,
        },
    )

    _remove_stale_runs(directory, keep_run_id=run_id)
    logger.info(
        "Saved clustering run %s for %s: %d clusters, %d noise documents, topics=%s",
        run_id,
        dataset,
        n_clusters,
        n_noise,
        extraction.method,
    )

    return ClusterRun(
        dataset=dataset,
        slug=dataset_dir(data_dir, dataset).name,
        run_id=run_id,
        timestamp=timestamp,
        params=params,
        index_fingerprint=index_fingerprint,
        pipeline_version=PIPELINE_VERSION,
        doc_ids=doc_ids,
        cluster_labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        topics_available=extraction.available,
        topics_method=extraction.method,
        warnings=warnings,
        embeddings_2d=embeddings_2d,
        topics=topics,
    )


def _remove_stale_runs(directory: Path, *, keep_run_id: str) -> None:
    """Delete derived files of previous runs (and leftovers of interrupted ones), plus the pre-run_id layout."""
    keep_prefix = f"run-{keep_run_id}-"
    legacy = {"topics.json", "embeddings_2d.npy"}
    for item in directory.iterdir():
        if not item.is_file() or item.name == MANIFEST_NAME:
            continue
        if item.name in legacy or (item.name.startswith("run-") and not item.name.startswith(keep_prefix)):
            try:
                item.unlink()
            except OSError as exc:  # pragma: no cover - only a permissions problem is possible here
                logger.warning("Could not remove stale clustering artifact %s (%s)", item, exc)


def _load_topics(path: Path) -> Dict[int, TopicInfo]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(cluster_id): TopicInfo(**info) for cluster_id, info in raw.items()}


def _load_projection(path: Path, n_documents: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    projection = np.load(path)
    if projection.ndim != 2 or projection.shape[0] != n_documents:
        logger.warning("Ignoring %s: shape %s does not match %d documents", path, projection.shape, n_documents)
        return None
    return projection


def load(data_dir: Path, dataset: str) -> Optional[ClusterRun]:
    """The stored run, or None when the dataset was never clustered (or its manifest is unreadable)."""
    directory = dataset_dir(data_dir, dataset)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            logger.warning(
                "Clusters for %s use layout version %s (this service writes %s); re-run clustering",
                dataset,
                manifest.get("schema_version"),
                SCHEMA_VERSION,
            )
            return None
        run_id = manifest["run_id"]
        params = ClusterParams(**manifest["params"])
        cluster_labels = [int(label) for label in manifest["cluster_labels"]]
        doc_ids = [str(doc_id) for doc_id in manifest["doc_ids"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("Unreadable clustering manifest %s (%s); re-run clustering", manifest_path, exc)
        return None

    topics = _load_topics(run_artifact(directory, run_id, TOPICS_KIND))
    projection = _load_projection(run_artifact(directory, run_id, EMBEDDINGS_2D_KIND), len(cluster_labels))

    return ClusterRun(
        dataset=manifest.get("dataset", dataset),
        slug=directory.name,
        run_id=run_id,
        timestamp=manifest.get("timestamp", ""),
        params=params,
        index_fingerprint=manifest.get("index_fingerprint", ""),
        pipeline_version=int(manifest.get("pipeline_version", 0)),
        doc_ids=doc_ids,
        cluster_labels=cluster_labels,
        n_clusters=int(manifest.get("n_clusters", len(set(cluster_labels) - {-1}))),
        n_noise=int(manifest.get("n_noise", sum(1 for label in cluster_labels if label == -1))),
        topics_available=bool(manifest.get("topics_available", bool(topics))),
        topics_method=manifest.get("topics_method"),
        warnings=list(manifest.get("warnings") or []),
        embeddings_2d=projection,
        topics=topics,
    )


def read_summary(data_dir: Path, dataset: str) -> Optional[ClusterSummary]:
    """The summary of a stored run, or None when there is none (or it belongs to another pipeline)."""
    directory = dataset_dir(data_dir, dataset)
    summary_path = directory / SUMMARY_NAME
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        return None

    if not summary_path.exists():
        # A run written before summaries existed: rebuild the summary from the manifest.
        run = load(data_dir, dataset)
        if run is None:
            return None
        return ClusterSummary(
            dataset=run.dataset,
            timestamp=run.timestamp,
            params=run.params,
            index_fingerprint=run.index_fingerprint,
            pipeline_version=run.pipeline_version,
            n_documents=run.n_documents,
            n_clusters=run.n_clusters,
            n_noise=run.n_noise,
            topics_available=run.topics_available,
        )

    try:
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("schema_version") != SCHEMA_VERSION:
            return None
        return ClusterSummary(
            dataset=summary["dataset"],
            timestamp=summary.get("timestamp", ""),
            params=ClusterParams(**summary["params"]),
            index_fingerprint=summary.get("index_fingerprint", ""),
            pipeline_version=int(summary.get("pipeline_version", 0)),
            n_documents=int(summary.get("n_documents", 0)),
            n_clusters=int(summary.get("n_clusters", 0)),
            n_noise=int(summary.get("n_noise", 0)),
            topics_available=bool(summary.get("topics_available", False)),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("Unreadable clustering summary %s (%s)", summary_path, exc)
        return None


def delete(data_dir: Path, dataset: str) -> bool:
    """Remove every file of the dataset's clustering run. False when there was nothing to delete."""
    import shutil

    directory = dataset_dir(data_dir, dataset)
    if not directory.is_dir():
        return False
    shutil.rmtree(directory)
    logger.info("Deleted clustering data for dataset %s (%s)", dataset, directory)
    return True


def clustered_datasets(data_dir: Path) -> List[str]:
    """Dataset names with a readable manifest, sorted (the old code returned directory slugs)."""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    datasets: List[str] = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        summary = read_summary(root, item.name)
        if summary is not None:
            datasets.append(summary.dataset)
    return datasets

