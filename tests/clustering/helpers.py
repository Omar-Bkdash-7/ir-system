"""Deterministic stand-ins for HDBSCAN, BERTopic, UMAP and PostgreSQL.

The service takes every heavy collaborator through its constructor, so these tests exercise the real orchestration
(cache keys, locking, graceful degradation, storage) without clustering 1400 documents, downloading a topic model or
needing a database. Embeddings, though, are real files: the fingerprint logic is about files on disk.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from services.clustering.config import ClusteringSettings
from services.clustering.core.texts import DocumentTexts
from services.clustering.core.topics import TopicExtraction
from services.clustering.schemas import TopicInfo
from services.clustering.service import ClusteringService

DOC_TEXTS = [
    "wing aerodynamics lift drag boundary layer",
    "turbulent boundary layer skin friction heating",
    "supersonic flow shock wave expansion fan",
    "hypersonic vehicle thermal protection ablation",
]


def dataset_dir_name(dataset: str) -> str:
    return dataset.replace("/", "_").replace("-", "_").replace(".", "_")


def write_embedding_index(
    indices_dir: Path,
    dataset: str,
    embeddings: np.ndarray,
    doc_ids: Sequence[str],
) -> Path:
    base = Path(indices_dir) / dataset_dir_name(dataset) / "embedding"
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "embeddings.npy", embeddings)
    (base / "doc_id_map.json").write_text(json.dumps(list(doc_ids)), encoding="utf-8")
    return base


def two_cluster_embeddings(per_cluster: int = 5, dimension: int = 8, seed: int = 0):
    """Two well separated blobs plus doc ids/texts, so HDBSCAN-free fakes still get realistic data."""
    rng = np.random.default_rng(seed)
    first = rng.normal(0.0, 0.05, (per_cluster, dimension))
    second = rng.normal(2.0, 0.05, (per_cluster, dimension))
    embeddings = np.vstack([first, second]).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(len(embeddings))]
    texts = [DOC_TEXTS[i % len(DOC_TEXTS)] + f" study {i}" for i in range(len(embeddings))]
    labels = np.array([0] * per_cluster + [1] * per_cluster, dtype=np.int64)
    return embeddings, doc_ids, texts, labels


class FakeClusterer:
    """Records its calls and returns preset labels (default: one cluster, no noise)."""

    def __init__(self, labels: Optional[np.ndarray] = None) -> None:
        self.labels = labels
        self.calls = 0
        self.kwargs: Dict[str, object] = {}

    def __call__(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        self.calls += 1
        self.kwargs = kwargs
        if self.labels is not None:
            return np.asarray(self.labels, dtype=np.int64)
        return np.zeros(len(embeddings), dtype=np.int64)


class FakeTextLoader:
    def __init__(self, texts: Optional[Sequence[str]] = None, missing: int = 0, error: Optional[Exception] = None):
        self.texts = texts
        self.missing = missing
        self.error = error
        self.calls = 0

    def __call__(self, dataset: str, doc_ids: Sequence[str]) -> DocumentTexts:
        self.calls += 1
        if self.error is not None:
            raise self.error
        texts = list(self.texts) if self.texts is not None else [f"text about aerodynamics {i}" for i in range(len(doc_ids))]
        if len(texts) < len(doc_ids):
            texts = texts + [""] * (len(doc_ids) - len(texts))
        return DocumentTexts(texts=texts, missing=self.missing)


class FakeTopicExtractor:
    def __init__(self) -> None:
        self.calls = 0
        self.last: Dict[str, object] = {}

    def __call__(self, texts, embeddings, labels, **kwargs) -> TopicExtraction:
        self.calls += 1
        self.last = kwargs
        topics = {
            int(cluster_id): TopicInfo(topic_id=int(cluster_id), keywords=["wing", "flow"], representative_docs=[texts[0]])
            for cluster_id in sorted(set(int(label) for label in labels) - {-1})
        }
        return TopicExtraction(topics=topics, method="ctfidf")


class FakeProjector:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, embeddings, **kwargs) -> np.ndarray:
        self.calls += 1
        return np.zeros((len(embeddings), 2), dtype=np.float32)


@dataclass
class Harness:
    """A service over a temporary index and cluster directory, with every heavy step faked."""

    settings: ClusteringSettings
    embeddings: np.ndarray
    doc_ids: List[str]
    texts: List[str]
    labels: np.ndarray
    clusterer: FakeClusterer = field(default_factory=FakeClusterer)
    text_loader: FakeTextLoader = field(default_factory=FakeTextLoader)
    topic_extractor: FakeTopicExtractor = field(default_factory=FakeTopicExtractor)
    projector: FakeProjector = field(default_factory=FakeProjector)
    dataset: str = "cranfield"
    _run_ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def service(self, **overrides) -> ClusteringService:
        options = dict(
            settings=self.settings,
            text_loader=self.text_loader,
            clusterer=self.clusterer,
            topic_extractor=self.topic_extractor,
            projector=self.projector,
            availability_check=lambda: None,
            run_id_factory=lambda: f"run{next(self._run_ids):04d}",
        )
        options.update(overrides)
        return ClusteringService(**options)

    def run(self, service: Optional[ClusteringService] = None, **kwargs):
        return (service or self.service()).cluster_dataset(self.dataset, **kwargs)

    def rewrite_index(self, scale: float = 2.0) -> None:
        """Change the embedding file on disk so its size/mtime (and therefore the fingerprint) differ."""
        time.sleep(0.01)
        write_embedding_index(
            self.settings.indices_dir,
            self.dataset,
            self.embeddings * scale,
            self.doc_ids,
        )


def make_harness(tmp_path: Path, *, with_index: bool = True, **settings_overrides) -> Harness:
    embeddings, doc_ids, texts, labels = two_cluster_embeddings()
    settings = ClusteringSettings(
        data_dir=Path(tmp_path) / "clusters",
        indices_dir=Path(tmp_path) / "indices",
        **settings_overrides,
    )
    if with_index:
        write_embedding_index(settings.indices_dir, "cranfield", embeddings, doc_ids)
    return Harness(
        settings=settings,
        embeddings=embeddings,
        doc_ids=list(doc_ids),
        texts=list(texts),
        labels=labels,
        clusterer=FakeClusterer(labels),
        text_loader=FakeTextLoader(texts),
    )
