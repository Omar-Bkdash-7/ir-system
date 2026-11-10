"""HTTP API of the clustering & topic detection service.

Failures are typed (services/clustering/errors.py) and translated to status codes here; no route wraps its body in a
catch-all, which is what used to turn "no clusters yet" into a 500. Dataset ids are path parameters, so ids that
contain '/' (beir/quora) are routed to the same handler instead of 404-ing on the path.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional, Type

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
from services.clustering.config import get_settings
from services.clustering.core.clusterer import CLUSTERING_AVAILABLE
from services.clustering.errors import (
    ClusteringError,
    ClusteringInProgressError,
    ClusteringUnavailableError,
    ClustersNotFoundError,
    DocumentNotClusteredError,
    EmbeddingIndexMissingError,
    InvalidClusteringParameterError,
)
from services.clustering.schemas import (
    ClusterListResponse,
    ClusterRequest,
    ClusterResponse,
    ClusterStatusResponse,
    DocumentClusterInfo,
)
from services.clustering.service import ClusteringService

logger = setup_logging("clustering")

#: Typed failure -> HTTP status. 404/400/409 are answers, not crashes, and each one keeps its own detail message.
ERROR_STATUS: Dict[Type[ClusteringError], int] = {
    ClusteringUnavailableError: 503,
    ClusteringInProgressError: 409,
    InvalidClusteringParameterError: 400,
    EmbeddingIndexMissingError: 404,
    ClustersNotFoundError: 404,
    DocumentNotClusteredError: 404,
}


def _error_handler(status_code: int):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        logger.warning("%s -> %s: %s", exc.__class__.__name__, status_code, exc)
        return JSONResponse(
            status_code=status_code,
            content={"error": exc.__class__.__name__, "detail": str(exc)},
        )

    return handler


def create_app(service: Optional[ClusteringService] = None) -> FastAPI:
    """Build the app; `service` is injected by the tests (no HDBSCAN, no database)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        logger.info("Clustering service starting (data_dir=%s)", settings.data_dir)
        if not CLUSTERING_AVAILABLE:
            logger.error(
                "hdbscan/umap are missing from this image: POST /cluster will answer 503 "
                "(stored runs can still be read)"
            )
        yield
        logger.info("Clustering service stopped")

    app = FastAPI(title="Clustering & Topic Detection Service", lifespan=lifespan)
    register_exception_handlers(app)
    for error_class, status_code in ERROR_STATUS.items():
        app.add_exception_handler(error_class, _error_handler(status_code))

    clustering_service = service if service is not None else ClusteringService()
    app.state.clustering_service = clustering_service

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "clustering", "clustering_available": CLUSTERING_AVAILABLE}

    @app.post("/cluster", response_model=ClusterResponse)
    def cluster_dataset(req: ClusterRequest):
        """Cluster the dataset's embeddings with HDBSCAN, detect topics, and store the run."""
        logger.info(
            "Clustering request for dataset=%s min_cluster_size=%s min_samples=%s metric=%s force=%s",
            req.dataset,
            req.min_cluster_size,
            req.min_samples,
            req.metric,
            req.force_recompute,
        )
        return clustering_service.cluster_dataset(
            req.dataset,
            min_cluster_size=req.min_cluster_size,
            min_samples=req.min_samples,
            metric=req.metric,
            n_keywords=req.n_keywords,
            force_recompute=req.force_recompute,
            include_visualization=req.include_visualization,
        )

    @app.get("/clusters/{dataset:path}", response_model=ClusterResponse)
    def get_clusters(
        dataset: str,
        include_visualization: bool = Query(True, description="Include the 2D projection (large for big datasets)"),
    ):
        """Stored cluster assignments and topics for a dataset."""
        return clustering_service.get_clusters(dataset, include_visualization=include_visualization)

    @app.get("/document/{dataset:path}/{doc_id}", response_model=DocumentClusterInfo)
    def get_document_cluster(dataset: str, doc_id: str):
        """Cluster and topic of one document."""
        return clustering_service.get_document_cluster(dataset, doc_id)

    @app.get("/datasets", response_model=ClusterListResponse)
    def list_clustered_datasets():
        """Datasets that already have a stored clustering run."""
        return ClusterListResponse(datasets=clustering_service.list_clustered_datasets())

    @app.delete("/clusters/{dataset:path}")
    def delete_clusters(dataset: str):
        """Delete the stored run of a dataset."""
        clustering_service.delete_clusters(dataset)
        return {"status": "success", "message": f"Clusters deleted for dataset {dataset}"}

    @app.get("/status/{dataset:path}", response_model=ClusterStatusResponse)
    def get_status(dataset: str):
        """Whether the dataset is clustered, with which parameters, and whether the embedding index changed since."""
        return clustering_service.get_status(dataset)

    return app


app = create_app()
