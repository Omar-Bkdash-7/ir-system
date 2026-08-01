from fastapi import FastAPI, HTTPException, Body, Query
from typing import Optional
from services.indexing.dataset_service import (
    validate_and_store_metadata,
    list_local_datasets,
    delete_local_dataset,
    upload_dataset_to_db,
    delete_dataset_from_db,
    get_db_dataset_info,
)
from services.indexing.indexing_service import (
    get_index_stats,
    build_tfidf_index,
    get_tfidf_status,
    build_embedding_index,
    get_embedding_status,
    build_bm25_index,
    get_bm25_status
)
from services.indexing.services.inverted_index_service import (
    get_inverted_index_stats,
    get_term_info,
    search_terms,
    get_vocabulary_sample
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
        import ir_datasets
        logger.info(f"Attempting to load dataset: {dataset_id}")
        
        # This will download the dataset if not present
        ds = ir_datasets.load(dataset_id)
        
        # Force iteration to ensure all components are downloaded
        doc_count = 0
        query_count = 0
        qrel_count = 0
        
        # Count documents (sample first 10 to verify)
        if hasattr(ds, 'docs_iter'):
            for i, _ in enumerate(ds.docs_iter()):
                doc_count += 1
                if i >= 10:  # Just verify first few docs
                    break
        
        # Count queries
        if hasattr(ds, 'queries_iter'):
            query_count = sum(1 for _ in ds.queries_iter())
        
        # Count qrels
        if hasattr(ds, 'qrels_iter'):
            qrel_count = sum(1 for _ in ds.qrels_iter())
        
        logger.info(f"Dataset {dataset_id} loaded: docs_sampled={doc_count}, queries={query_count}, qrels={qrel_count}")
        
        return {
            "status": "success",
            "dataset_id": dataset_id,
            "message": "Dataset downloaded and verified",
            "sample_counts": {
                "docs_sampled": doc_count,
                "queries": query_count,
                "qrels": qrel_count
            }
        }
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
        from services.indexing.dataset.path_resolver import DatasetPathResolver
        import os
        
        resolver = DatasetPathResolver()
        ir_home = resolver.get_ir_home()
        
        result = {
            "ir_datasets_home": str(ir_home),
            "exists": ir_home.exists(),
            "env_var": os.environ.get("IR_DATASETS_HOME"),
        }
        
        if ir_home.exists():
            children = list(ir_home.iterdir())
            result["contents"] = [
                {
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "path": str(child)
                }
                for child in sorted(children)
            ]
            
            # Check beir/quora specifically
            quora_path = ir_home / "beir" / "quora"
            if quora_path.exists():
                result["beir_quora"] = {
                    "exists": True,
                    "contents": [f.name for f in quora_path.iterdir()]
                }
            else:
                result["beir_quora"] = {"exists": False}
        
        return result
    except Exception as e:
        logger.error(f"Debug path failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/datasets/local")
def delete_dataset(dataset_id: str = Query(..., description="Dataset ID to delete")):
    """Permanently delete a local dataset directory."""
    try:
        return delete_local_dataset(dataset_id)
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
    except Exception as e:
        logger.error(f"Upload to DB failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/db-info")
def db_info():
    """Return row count from the documents table."""
    return get_db_dataset_info()


@app.delete("/datasets/db")
def delete_dataset_from_db_endpoint(dataset_id: str = Query(..., description="Dataset ID to delete from DB")):
    """Delete all documents for a specific dataset from PostgreSQL."""
    try:
        return delete_dataset_from_db(dataset_id)
    except Exception as e:
        logger.error(f"Delete from DB failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Index management ──────────────────────────────────────────────────────

@app.get("/index/stats")
def index_stats():
    return get_index_stats()


@app.post("/index/tfidf/build")
def build_tfidf(
    force_rebuild: bool = Query(False, description="Force rebuild the index"), 
    body: Optional[dict] = Body(None)
):
    preprocessing_config = body.get("preprocessing_config") if body else None
    return build_tfidf_index(force_rebuild, preprocessing_config)


@app.get("/index/tfidf/status")
def tfidf_status():
    return get_tfidf_status()


@app.post("/index/embedding/build")
def build_embedding(
    force_rebuild: bool = Query(False, description="Force rebuild the index"),
    body: Optional[dict] = Body(None)
):
    preprocessing_config = body.get("preprocessing_config") if body else None
    return build_embedding_index(force_rebuild, preprocessing_config)


@app.get("/index/embedding/status")
def embedding_status():
    return get_embedding_status()


@app.post("/index/bm25/build")
def build_bm25(
    force_rebuild: bool = Query(False, description="Force rebuild the index"),
    body: Optional[dict] = Body(None)
):
    preprocessing_config = body.get("preprocessing_config") if body else None
    return build_bm25_index(force_rebuild, preprocessing_config)


@app.get("/index/bm25/status")
def bm25_status():
    return get_bm25_status()


# ── Inverted Index Analysis ───────────────────────────────────────────────

@app.get("/index/inverted/stats")
def inverted_stats():
    """Get comprehensive inverted index statistics."""
    return get_inverted_index_stats()


@app.get("/index/inverted/term/{term}")
def term_info(term: str):
    """Get detailed information about a specific term."""
    return get_term_info(term)


@app.get("/index/inverted/search")
def search_vocabulary(
    prefix: str = Query(..., description="Prefix to search for"),
    limit: int = Query(50, description="Maximum number of results")
):
    """Search for terms starting with given prefix."""
    return search_terms(prefix, limit)


@app.get("/index/inverted/sample")
def vocabulary_sample(sample_size: int = Query(100, description="Sample size")):
    """Get a random sample of vocabulary terms."""
    return get_vocabulary_sample(sample_size)
