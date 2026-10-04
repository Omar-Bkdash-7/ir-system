from fastapi import FastAPI, HTTPException, Query
from services.clustering.clustering_service import ClusteringService
from services.clustering.schemas import (
    ClusterRequest,
    TopicRequest,
    ClusterResponse,
    DocumentClusterInfo,
    ClusterListResponse,
)
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
import logging

logger = setup_logging("clustering")

app = FastAPI(title="Clustering & Topic Detection Service")
register_exception_handlers(app)

clustering_service = ClusteringService()


@app.on_event("startup")
def startup_event():
    logger.info("Clustering & Topic Detection service starting up...")
    logger.info("Service ready for clustering and topic modeling")


@app.get("/health")
def health():
    return {"status": "ok", "service": "clustering"}


@app.post("/cluster", response_model=ClusterResponse)
def cluster_dataset(req: ClusterRequest):
    """
    Cluster documents in a dataset using HDBSCAN on embeddings.
    Also performs topic detection using BERTopic.
    """
    logger.info(
        f"Clustering request for dataset: {req.dataset} "
        f"(min_cluster_size={req.min_cluster_size}, min_samples={req.min_samples})"
    )
    
    try:
        result = clustering_service.cluster_dataset(
            dataset=req.dataset,
            min_cluster_size=req.min_cluster_size,
            min_samples=req.min_samples,
            metric=req.metric,
            force_recompute=req.force_recompute,
        )
        logger.info(
            f"Clustering complete: {result.n_clusters} clusters, {result.n_noise} noise points"
        )
        return result
    except RuntimeError as e:
        # Handle missing libraries error
        error_msg = str(e)
        logger.error(f"Runtime error: {error_msg}")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail=error_msg,
        )
    except FileNotFoundError as e:
        logger.error(f"Embeddings not found for dataset {req.dataset}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Embeddings not found for dataset {req.dataset}. Build embeddings first via indexing service.",
        )
    except Exception as e:
        logger.error(f"Error clustering dataset {req.dataset}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clusters/{dataset}", response_model=ClusterResponse)
def get_clusters(dataset: str):
    """
    Get cluster assignments and topics for a dataset.
    """
    logger.info(f"Getting clusters for dataset: {dataset}")
    
    try:
        result = clustering_service.get_clusters(dataset)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Clusters not found for dataset {dataset}. Run clustering first.",
            )
        return result
    except Exception as e:
        logger.error(f"Error getting clusters for dataset {dataset}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{dataset}/{doc_id}", response_model=DocumentClusterInfo)
def get_document_cluster(dataset: str, doc_id: str):
    """
    Get cluster and topic information for a specific document.
    """
    logger.info(f"Getting cluster info for doc {doc_id} in dataset {dataset}")
    
    try:
        result = clustering_service.get_document_cluster(dataset, doc_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id} not found or clusters not computed for dataset {dataset}",
            )
        return result
    except Exception as e:
        logger.error(
            f"Error getting cluster for doc {doc_id} in dataset {dataset}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets", response_model=ClusterListResponse)
def list_clustered_datasets():
    """
    List all datasets that have been clustered.
    """
    logger.info("Listing clustered datasets")
    
    try:
        datasets = clustering_service.list_clustered_datasets()
        return ClusterListResponse(datasets=datasets)
    except Exception as e:
        logger.error(f"Error listing clustered datasets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clusters/{dataset}")
def delete_clusters(dataset: str):
    """
    Delete cluster data for a dataset.
    """
    logger.info(f"Deleting clusters for dataset: {dataset}")
    
    try:
        success = clustering_service.delete_clusters(dataset)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Clusters not found for dataset {dataset}",
            )
        return {"status": "success", "message": f"Clusters deleted for dataset {dataset}"}
    except Exception as e:
        logger.error(f"Error deleting clusters for dataset {dataset}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{dataset}")
def get_status(dataset: str):
    """
    Get clustering status for a dataset.
    """
    logger.info(f"Getting status for dataset: {dataset}")
    
    try:
        status = clustering_service.get_status(dataset)
        return status
    except Exception as e:
        logger.error(f"Error getting status for dataset {dataset}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
