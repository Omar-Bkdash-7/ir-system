"""Clustering + topic detection for a dataset's embedding index.

Flow of POST /cluster:
    validate dataset/parameters -> cheap cache check (params + embedding-index fingerprint, no matrix read)
    -> cluster (HDBSCAN) -> topics (BERTopic, sklearn fallback, or none) -> 2D projection -> atomic save.

Everything expensive is injectable (index loader, text loader, clusterer, topic extractor, projector) so the tests
exercise the orchestration without hdbscan/bertopic/torch or a database.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from services.clustering.config import ClusteringSettings, get_settings
from services.clustering.core import store
from services.clustering.core.clusterer import project_2d, require_clustering_available, run_hdbscan, validate_parameters
from services.clustering.core.embeddings import (
    EmbeddingIndex,
    index_fingerprint,
    load_embedding_index,
)
from services.clustering.core.texts import DocumentTexts, fetch_texts
from services.clustering.core.topics import TopicExtraction, extract_topics
from services.clustering.errors import (
    ClusteringInProgressError,
    ClustersNotFoundError,
    DocumentNotClusteredError,
    EmbeddingIndexMissingError,
    TextsUnavailableError,
)
from services.clustering.schemas import (
    ClusterInfo,
    ClusterParams,
    ClusterResponse,
    ClusterStatusResponse,
    DocumentClusterInfo,
)
from services.indexing.storage import try_lock

logger = logging.getLogger(__name__)


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"


class ClusteringService:
    def __init__(
        self,
        settings: Optional[ClusteringSettings] = None,
        *,
        index_loader: Callable[[str], EmbeddingIndex] = None,
        fingerprint_loader: Callable[[str], str] = None,
        text_loader: Callable[[str, Sequence[str]], DocumentTexts] = None,
        clusterer: Callable[..., np.ndarray] = None,
        topic_extractor: Callable[..., TopicExtraction] = None,
        projector: Callable[..., np.ndarray] = None,
        availability_check: Callable[[], None] = None,
        run_id_factory: Callable[[], str] = _new_run_id,
    ) -> None:
        self.settings = settings or get_settings()
        self._availability_check = availability_check or require_clustering_available
        self._index_loader = index_loader or (lambda dataset: load_embedding_index(self.settings.indices_dir, dataset))
        self._fingerprint_loader = fingerprint_loader or (
            lambda dataset: index_fingerprint(self.settings.indices_dir, dataset)
        )
        self._text_loader = text_loader or self._load_texts_from_database
        self._clusterer = clusterer or run_hdbscan
        self._topic_extractor = topic_extractor or extract_topics
        self._projector = projector or project_2d
        self._new_run_id = run_id_factory
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ClusteringService ready (data=%s, indices=%s, default min_cluster_size=%d, metric=%s)",
            self.settings.data_dir,
            self.settings.indices_dir,
            self.settings.min_cluster_size,
            self.settings.metric,
        )

    # ------------------------------------------------------------------ public API

    def cluster_dataset(
        self,
        dataset: str,
        *,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        metric: Optional[str] = None,
        n_keywords: Optional[int] = None,
        force_recompute: bool = False,
        include_visualization: bool = True,
    ) -> ClusterResponse:
        params = self._resolve_params(min_cluster_size, min_samples, metric, n_keywords)
        fingerprint = self._fingerprint_loader(dataset)
        cached = store.load(self.settings.data_dir, dataset)

        if not force_recompute and cached is not None and cached.matches(params, fingerprint, store.PIPELINE_VERSION):
            logger.info(
                "Serving the stored clustering run %s for %s (params and embedding index unchanged)",
                cached.run_id,
                dataset,
            )
            return self._build_response(cached, include_visualization)

        if cached is not None and not force_recompute:
            if cached.pipeline_version != store.PIPELINE_VERSION:
                reason = f"stored run uses pipeline version {cached.pipeline_version}"
            elif cached.index_fingerprint != fingerprint:
                reason = "embedding index changed"
            else:
                reason = "parameters changed"
            logger.info("Recomputing clusters for %s (%s)", dataset, reason)

        # Libraries are checked before taking the lock so a 503 never blocks a concurrent valid run.
        self._availability_check()

        with try_lock(f"clustering:{dataset}") as acquired:
            if not acquired:
                raise ClusteringInProgressError(f"A clustering run for dataset {dataset!r} is already in progress")

            started = time.perf_counter()
            index = self._index_loader(dataset)
            validate_parameters(
                min_cluster_size=params.min_cluster_size,
                min_samples=params.min_samples,
                metric=params.metric,
                n_keywords=params.n_keywords,
                n_docs=index.n_docs,
            )
            logger.info(
                "Clustering %s: %d documents, dimension %d, params=%s",
                dataset,
                index.n_docs,
                index.dimension,
                params.model_dump(),
            )

            labels = self._clusterer(
                index.embeddings,
                min_cluster_size=params.min_cluster_size,
                min_samples=params.min_samples,
                metric=params.metric,
                random_state=self.settings.random_state,
            )

            warnings: List[str] = []
            extraction = self._extract_topics(dataset, index, labels, params, warnings)

            embeddings_2d = None
            if include_visualization:
                embeddings_2d = self._projector(
                    index.embeddings,
                    n_neighbors=self.settings.umap_n_neighbors,
                    min_dist=self.settings.umap_min_dist,
                    metric=self.settings.umap_metric,
                    random_state=self.settings.random_state,
                )

            run = store.save(
                self.settings.data_dir,
                dataset,
                run_id=self._new_run_id(),
                params=params,
                index_fingerprint=index.fingerprint,
                doc_ids=index.doc_ids,
                cluster_labels=labels,
                extraction=extraction,
                embeddings_2d=embeddings_2d,
                warnings=warnings,
            )

        logger.info(
            "Clustering %s finished in %.1fs: %d clusters, %d noise documents",
            dataset,
            time.perf_counter() - started,
            run.n_clusters,
            run.n_noise,
        )
        return self._build_response(run, include_visualization)

    def get_clusters(self, dataset: str, include_visualization: bool = True) -> ClusterResponse:
        run = store.load(self.settings.data_dir, dataset)
        if run is None:
            raise ClustersNotFoundError(
                f"Clusters not found for dataset {dataset!r}. Run clustering first (POST /cluster)."
            )
        return self._build_response(run, include_visualization)

    def get_document_cluster(self, dataset: str, doc_id: str) -> DocumentClusterInfo:
        run = store.load(self.settings.data_dir, dataset)
        if run is None:
            raise ClustersNotFoundError(
                f"Clusters not found for dataset {dataset!r}. Run clustering first (POST /cluster)."
            )
        position = run.document_index().get(str(doc_id))
        if position is None:
            raise DocumentNotClusteredError(
                f"Document {doc_id!r} is not part of the clustering run of dataset {dataset!r}"
            )
        cluster_id = int(run.cluster_labels[position])
        return DocumentClusterInfo(
            doc_id=str(doc_id),
            cluster_id=cluster_id,
            topic=run.topics.get(cluster_id),
        )

    def list_clustered_datasets(self) -> List[str]:
        return store.clustered_datasets(self.settings.data_dir)

    def delete_clusters(self, dataset: str) -> None:
        if not store.delete(self.settings.data_dir, dataset):
            raise ClustersNotFoundError(f"Clusters not found for dataset {dataset!r}")

    def get_status(self, dataset: str) -> ClusterStatusResponse:
        # The summary file only: /status is polled by the UI and must not read the label arrays (or the projection).
        summary = store.read_summary(self.settings.data_dir, dataset)
        if summary is None:
            return ClusterStatusResponse(
                dataset=dataset,
                status="not_clustered",
                message="Dataset has not been clustered yet",
            )

        stale = summary.pipeline_version != store.PIPELINE_VERSION or self._is_stale(
            dataset, summary.index_fingerprint
        )
        return ClusterStatusResponse(
            dataset=summary.dataset,
            status="clustered",
            n_clusters=summary.n_clusters,
            n_noise=summary.n_noise,
            n_documents=summary.n_documents,
            timestamp=summary.timestamp,
            params=summary.params,
            topics_available=summary.topics_available,
            stale=stale,
        )

    # ------------------------------------------------------------------ helpers

    def _resolve_params(
        self,
        min_cluster_size: Optional[int],
        min_samples: Optional[int],
        metric: Optional[str],
        n_keywords: Optional[int],
    ) -> ClusterParams:
        """Request values win; anything left None comes from the configuration."""
        return ClusterParams(
            min_cluster_size=int(min_cluster_size if min_cluster_size is not None else self.settings.min_cluster_size),
            min_samples=int(min_samples if min_samples is not None else self.settings.min_samples),
            metric=(metric or self.settings.metric),
            n_keywords=int(n_keywords if n_keywords is not None else self.settings.n_keywords),
        )

    def _is_stale(self, dataset: str, run_fingerprint: str) -> bool:
        try:
            return self._fingerprint_loader(dataset) != run_fingerprint
        except EmbeddingIndexMissingError:
            return True  # the index behind these clusters is gone: they cannot be reproduced
        except Exception as exc:  # noqa: BLE001 - status must answer even when the check itself fails
            logger.warning("Could not check whether the clusters of %s are stale (%s)", dataset, exc)
            return False

    def _extract_topics(
        self,
        dataset: str,
        index: EmbeddingIndex,
        labels: np.ndarray,
        params: ClusterParams,
        warnings: List[str],
    ) -> TopicExtraction:
        """Topics are optional: missing texts or a failing extractor must not lose the clustering result."""
        try:
            texts = self._text_loader(dataset, index.doc_ids)
        except TextsUnavailableError as exc:
            logger.warning("Topics disabled for %s: %s", dataset, exc)
            warnings.append(f"topics unavailable: {exc}")
            return TopicExtraction(topics={}, method="none")

        if not texts.usable(self.settings.max_missing_text_ratio):
            message = (
                f"topics unavailable: {texts.missing} of {texts.total} documents have no text "
                f"(limit {self.settings.max_missing_text_ratio:.0%})"
            )
            logger.warning("Topics disabled for %s: %s", dataset, message)
            warnings.append(message)
            return TopicExtraction(topics={}, method="none")

        if texts.missing:
            warnings.append(f"{texts.missing} of {texts.total} documents have no text and were ignored for topics")

        try:
            return self._topic_extractor(
                texts.texts,
                index.embeddings,
                labels,
                n_keywords=params.n_keywords,
                representative_count=self.settings.representative_docs,
            )
        except Exception as exc:  # noqa: BLE001 - keyword extraction is best-effort by design
            logger.warning("Topic extraction failed for %s (%s); clusters were kept", dataset, exc)
            warnings.append(f"topics unavailable: {exc}")
            return TopicExtraction(topics={}, method="none")

    def _load_texts_from_database(self, dataset: str, doc_ids: Sequence[str]) -> DocumentTexts:
        from services.retrieval.core.texts import TextStore

        text_store = TextStore(pool_max=self.settings.db_pool_max)
        try:
            return fetch_texts(
                dataset,
                doc_ids,
                batch_size=self.settings.text_batch_size,
                store=text_store,
            )
        finally:
            text_store.close()

    def _build_response(self, run: store.ClusterRun, include_visualization: bool) -> ClusterResponse:
        sizes: Dict[int, int] = {}
        for label in run.cluster_labels:
            if label != -1:
                sizes[label] = sizes.get(label, 0) + 1

        clusters = [
            ClusterInfo(cluster_id=cluster_id, size=size, topic=run.topics.get(cluster_id))
            for cluster_id, size in sorted(sizes.items())
        ]
        projection = None
        if include_visualization and run.embeddings_2d is not None:
            projection = run.embeddings_2d.tolist()

        return ClusterResponse(
            dataset=run.dataset,
            n_clusters=run.n_clusters,
            n_noise=run.n_noise,
            n_documents=run.n_documents,
            clusters=clusters,
            embeddings_2d=projection,
            cluster_labels=run.cluster_labels,
            params=run.params,
            topics_available=run.topics_available,
            topics_method=run.topics_method,
            warnings=list(run.warnings),
        )
