from fastapi import FastAPI, HTTPException, Body, Query
from typing import Optional
from services.indexing.dataset import maintenance
from services.indexing.dataset import (
    validate_and_store_metadata,
    list_local_datasets,
    delete_local_dataset,
    upload_dataset_to_db,
    delete_dataset_from_db,
    get_db_dataset_info,
)
from services.indexing.errors import InvalidDatasetIdError, InvalidIndexTypeError
from services.indexing.services.build_service import (
    build_tfidf_index,
    build_embedding_index,
    build_bm25_index,
    build_all_indices,
)
from services.indexing.services.index_admin import (
    delete_all_indices,
    delete_index as delete_index_files,
)
from services.indexing.services.inverted_index_service import (
    get_inverted_index_stats,
    get_term_info,
    search_terms,
    get_vocabulary_sample,
)
from services.indexing.services.stats_service import (
    get_index_stats,
    get_tfidf_stats,
    get_embedding_stats,
    get_bm25_stats,
)
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging

logger = setup_logging("indexing")

app = FastAPI(title="Indexing Service")
register_exception_handlers(app)


@app.get("/health")
def health():
    return {"status": "ok", "service": "indexing"}


@app.post("/validate")
def validate_dataset_endpoint(dataset_id: str = Query(..., description="Dataset ID to validate")):
    """Validate a dataset and store its metadata."""
    return validate_and_store_metadata(dataset_id)


# ── Dataset management ────────────────────────────────────────────────────

@app.post("/datasets/download")
def download_dataset(dataset_id: str = Query(..., description="Dataset ID to download")):
    """
    Force download/verify dataset using ir_datasets library.
    This ensures the dataset is available in the container filesystem.
    """
    try:
        return maintenance.download_dataset(dataset_id)
    except Exception as e:
        logger.error(f"Failed to download dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/local")
def local_datasets():
    """List all locally downloaded datasets."""
    try:
        datasets = list_local_datasets()
        logger.info(f"Found {len(datasets)} local datasets")
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Error listing local datasets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/debug-path")
def debug_dataset_path():
    """Debug endpoint to inspect IR_DATASETS_HOME contents."""
    try:
        return maintenance.debug_path_info()
    except Exception as e:
        logger.error(f"Debug path failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/datasets/local")
def delete_dataset(dataset_id: str = Query(..., description="Dataset ID to delete")):
    """Permanently delete a local dataset directory."""
    try:
        return delete_local_dataset(dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Delete failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/upload-to-db")
def upload_to_db(
    dataset_id: str = Query(..., description="Dataset ID to upload"), 
    batch_size: int = Query(1000, description="Batch size for upload")
):
    """Load documents from a local ir_datasets dataset into PostgreSQL."""
    try:
        return upload_dataset_to_db(dataset_id, batch_size=batch_size)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload to DB failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/db-info")
def db_info():
    """Return row count from the documents table."""
    return get_db_dataset_info()


@app.get("/datasets/metadata")
def list_datasets_metadata():
    """List all datasets with their metadata."""
    try:
        from services.indexing.dataset.service import DatasetService
        service = DatasetService()
        datasets = service.list_datasets()
        
        return {
            "datasets": datasets,
            "count": len(datasets)
        }
    except Exception as e:
        logger.error(f"Failed to list datasets metadata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/datasets/db")
def delete_dataset_from_db_endpoint(dataset_id: str = Query(..., description="Dataset ID to delete from DB")):
    """Delete all documents for a specific dataset from PostgreSQL."""
    try:
        return delete_dataset_from_db(dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Delete from DB failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/fix-corrupted")
def fix_corrupted_dataset(dataset_id: str = Query(..., description="Dataset ID to fix")):
    """
    Fix corrupted dataset by deleting local files and re-downloading.

    This endpoint:
    1. Validates the dataset id (an empty or unsafe id is rejected with 400 and nothing is deleted)
    2. Deletes the corrupted local dataset
    3. Re-downloads it completely from ir_datasets registry
    """
    try:
        return maintenance.fix_corrupted_dataset(dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to fix corrupted dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Index management ──────────────────────────────────────────────────────

@app.get("/index/stats")
def index_stats(
    dataset_name: str = Query(..., description="Dataset name"),
    doc_count: Optional[int] = Query(None, description="Total number of documents to report (optional)"),
):
    return get_index_stats(dataset_name, doc_count)


@app.post("/index/tfidf/build")
def build_tfidf(
    dataset_name: str = Query(..., description="Dataset name to build index for"),
    force_rebuild: bool = Query(False, description="Force rebuild the index"), 
    body: Optional[dict] = Body(None)
):
    preprocessing_config = body.get("preprocessing_config") if body else None
    return build_tfidf_index(dataset_name, force_rebuild, preprocessing_config)


@app.get("/index/tfidf/status")
def tfidf_status(dataset_name: str = Query(..., description="Dataset name")):
    return get_tfidf_stats(dataset_name)


@app.post("/index/embedding/build")
def build_embedding(
    dataset_name: str = Query(..., description="Dataset name to build index for"),
    force_rebuild: bool = Query(False, description="Force rebuild the index"),
    body: Optional[dict] = Body(None)
):
    preprocessing_config = body.get("preprocessing_config") if body else None
    return build_embedding_index(dataset_name, force_rebuild, preprocessing_config)


@app.get("/index/embedding/status")
def embedding_status(dataset_name: str = Query(..., description="Dataset name")):
    return get_embedding_stats(dataset_name)


@app.post("/index/bm25/build")
def build_bm25(
    dataset_name: str = Query(..., description="Dataset name to build index for"),
    force_rebuild: bool = Query(False, description="Force rebuild the index"),
    body: Optional[dict] = Body(None)
):
    preprocessing_config = body.get("preprocessing_config") if body else None
    return build_bm25_index(dataset_name, force_rebuild, preprocessing_config)


@app.get("/index/bm25/status")
def bm25_status(dataset_name: str = Query(..., description="Dataset name")):
    return get_bm25_stats(dataset_name)


@app.post("/index/build-all")
def build_all(
    dataset_name: str = Query(..., description="Dataset name to build indices for"),
    force_rebuild: bool = Query(False, description="Force rebuild all indices"),
    body: Optional[dict] = Body(None)
):
    """
    Build all indices (TF-IDF, BM25, Embedding) with shared preprocessing.
    This is more efficient than building indices separately.
    """
    preprocessing_config = body.get("preprocessing_config") if body else None
    return build_all_indices(dataset_name, force_rebuild, preprocessing_config)


# ── Inverted Index Analysis ───────────────────────────────────────────────

@app.get("/index/inverted/stats")
def inverted_stats(dataset_name: str = Query(..., description="Dataset name")):
    """Get comprehensive inverted index statistics."""
    return get_inverted_index_stats(dataset_name)


@app.get("/index/inverted/term/{term}")
def term_info(term: str, dataset_name: str = Query(..., description="Dataset name")):
    """Get detailed information about a specific term."""
    return get_term_info(dataset_name, term)


@app.get("/index/inverted/search")
def search_vocabulary(
    prefix: str = Query(..., description="Prefix to search for"),
    limit: int = Query(50, description="Maximum number of results"),
    dataset_name: str = Query(..., description="Dataset name"),
):
    """Search for terms starting with given prefix."""
    return search_terms(dataset_name, prefix, limit)


@app.get("/index/inverted/sample")
def vocabulary_sample(
    sample_size: int = Query(100, description="Sample size"),
    dataset_name: str = Query(..., description="Dataset name"),
):
    """Get a random sample of vocabulary terms."""
    return get_vocabulary_sample(dataset_name, sample_size)


# ── Index Deletion ────────────────────────────────────────────────────────

@app.delete("/index/{index_type}/delete")
def delete_index(
    index_type: str,
    dataset_name: str = Query(..., description="Dataset name")
):
    """Delete a specific index type for a dataset."""
    try:
        return delete_index_files(index_type, dataset_name)
    except (InvalidIndexTypeError, InvalidDatasetIdError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete {index_type} index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index/delete-all")
def delete_all_indices_endpoint(dataset_name: str = Query(..., description="Dataset name")):
    """Delete all index files for a dataset."""
    try:
        return delete_all_indices(dataset_name)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete all indices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
