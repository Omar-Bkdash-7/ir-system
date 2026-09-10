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
    get_bm25_status,
    build_all_indices
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
    except Exception as e:
        logger.error(f"Delete from DB failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/fix-corrupted")
def fix_corrupted_dataset(dataset_id: str = Query(..., description="Dataset ID to fix")):
    """
    Fix corrupted dataset by deleting local files and re-downloading.
    
    This endpoint:
    1. Deletes the corrupted local dataset
    2. Re-downloads it completely from ir_datasets registry
    3. Validates the download
    """
    try:
        import shutil
        from services.indexing.dataset.path_resolver import DatasetPathResolver
        
        logger.info(f"Attempting to fix corrupted dataset: {dataset_id}")
        
        # Get dataset path
        resolver = DatasetPathResolver()
        ir_home = resolver.get_ir_home()
        
        parts = dataset_id.strip("/").split("/")
        dataset_path = ir_home
        for part in parts:
            dataset_path = dataset_path / part
        
        # Delete existing dataset
        if dataset_path.exists():
            logger.info(f"Deleting corrupted dataset at: {dataset_path}")
            shutil.rmtree(dataset_path)
            logger.info("Deleted corrupted dataset files")
        
        # Re-download dataset
        logger.info(f"Re-downloading dataset '{dataset_id}'...")
        ds = ir_datasets.load(dataset_id)
        
        # Force complete download by counting all items
        doc_count = 0
        if hasattr(ds, 'docs_iter'):
            logger.info("Iterating documents to force complete download...")
            for doc in ds.docs_iter():
                doc_count += 1
                if doc_count % 5000 == 0:
                    logger.info(f"Downloaded {doc_count:,} documents...")
        
        query_count = 0
        if hasattr(ds, 'queries_iter'):
            query_count = sum(1 for _ in ds.queries_iter())
        
        qrel_count = 0
        if hasattr(ds, 'qrels_iter'):
            qrel_count = sum(1 for _ in ds.qrels_iter())
        
        logger.info(
            f"Dataset fixed successfully: "
            f"docs={doc_count}, queries={query_count}, qrels={qrel_count}"
        )
        
        return {
            "status": "success",
            "dataset_id": dataset_id,
            "message": "Dataset fixed and re-downloaded successfully",
            "counts": {
                "documents": doc_count,
                "queries": query_count,
                "qrels": qrel_count
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to fix corrupted dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Index management ──────────────────────────────────────────────────────

@app.get("/index/stats")
def index_stats():
    return get_index_stats()


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
    return get_tfidf_status(dataset_name)


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
    return get_embedding_status(dataset_name)


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
    return get_bm25_status(dataset_name)


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


# ── Index Deletion ────────────────────────────────────────────────────────

@app.delete("/index/{index_type}/delete")
def delete_index(
    index_type: str,
    dataset_name: str = Query(..., description="Dataset name")
):
    """Delete a specific index type for a dataset."""
    from pathlib import Path
    
    if index_type not in ["tfidf", "bm25", "embedding"]:
        raise HTTPException(status_code=400, detail=f"Invalid index type: {index_type}")
    
    try:
        safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_').replace('.', '_')
        
        # The indices are stored in subdirectories: data/indices/{dataset}/{index_type}/
        indices_base = Path("data/indices")
        index_dir = indices_base / safe_dataset_name / index_type
        
        deleted_files = []
        
        if index_dir.exists() and index_dir.is_dir():
            # Delete all files in the index directory
            for file_path in index_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    deleted_files.append(file_path.name)
                    logger.info(f"Deleted: {file_path}")
            
            # Try to remove the directory itself if empty
            try:
                index_dir.rmdir()
                logger.info(f"Removed empty directory: {index_dir}")
            except OSError:
                pass  # Directory not empty or permission issue
        
        if deleted_files:
            return {
                "status": "success",
                "message": f"Deleted {index_type} index",
                "deleted_files": deleted_files,
                "count": len(deleted_files)
            }
        else:
            return {
                "status": "not_found",
                "message": f"No {index_type} index files found",
                "deleted_files": [],
                "count": 0
            }
    
    except Exception as e:
        logger.error(f"Failed to delete {index_type} index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index/delete-all")
def delete_all_indices_endpoint(dataset_name: str = Query(..., description="Dataset name")):
    """Delete all index files for a dataset."""
    from pathlib import Path
    import shutil
    
    try:
        safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_').replace('.', '_')
        
        # The dataset indices directory contains subdirectories for each index type
        dataset_indices_dir = Path("data/indices") / safe_dataset_name
        
        if dataset_indices_dir.exists() and dataset_indices_dir.is_dir():
            # Count files recursively
            file_count = sum(1 for _ in dataset_indices_dir.rglob('*') if _.is_file())
            
            # Delete the entire directory
            shutil.rmtree(dataset_indices_dir)
            
            logger.info(f"Deleted {file_count} index files for dataset: {dataset_name}")
            
            return {
                "status": "success",
                "message": f"Deleted all indices for {dataset_name}",
                "deleted_count": file_count
            }
        else:
            return {
                "status": "not_found",
                "message": "No index directory found",
                "deleted_count": 0
            }
    
    except Exception as e:
        logger.error(f"Failed to delete all indices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
