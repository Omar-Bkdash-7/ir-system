"""Indexing service: the HTTP layer. Routing and parameter validation only.

Everything the routes do lives in `services.indexing.core`, and every failure that
has a status (invalid id, missing dataset, failed build) is raised there as a typed
exception that the handlers below translate.
"""
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Body, Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
from services.indexing.core.errors import BuildFailedError
from services.indexing.core.service import IndexingService

logger = setup_logging("indexing")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = IndexingService.create()
    logger.info("Indexing service started")
    try:
        yield
    finally:
        logger.info("Indexing service stopped")


app = FastAPI(title="Indexing Service", lifespan=lifespan)
register_exception_handlers(app)


async def build_failed_handler(request: Request, exc: BuildFailedError) -> JSONResponse:
    """A build that reports `status: error` is an HTTP error, not a 200 with a failure body."""
    logger.error("Build failed on %s (%s): %s", request.url.path, exc.reason, exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


app.add_exception_handler(BuildFailedError, build_failed_handler)


def get_service(request: Request) -> IndexingService:
    service = getattr(request.app.state, "service", None)
    if service is None:  # e.g. a TestClient used without running the lifespan
        service = request.app.state.service = IndexingService.create()
    return service


def _preprocessing_config(body: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return body.get("preprocessing_config") if body else None


# ----------------------------------------------------------------------------------------------- health & datasets
@app.get("/health")
def health():
    return {"status": "ok", "service": "indexing"}


@app.post("/validate")
def validate_dataset_endpoint(
    dataset_id: str = Query(..., description="Dataset ID to validate"),
    service: IndexingService = Depends(get_service),
):
    """Validate a dataset and store its metadata."""
    return service.datasets.validate(dataset_id)


@app.post("/datasets/download")
def download_dataset(
    dataset_id: str = Query(..., description="Dataset ID to download"),
    service: IndexingService = Depends(get_service),
):
    """Force download/verify of a dataset with ir_datasets."""
    return service.datasets.download(dataset_id)


@app.get("/datasets/local")
def local_datasets(service: IndexingService = Depends(get_service)):
    """Every locally downloaded dataset."""
    return {"datasets": service.datasets.list_local()}


@app.delete("/datasets/local")
def delete_dataset(
    dataset_id: str = Query(..., description="Dataset ID to delete"),
    service: IndexingService = Depends(get_service),
):
    """Permanently delete a local dataset directory."""
    return service.datasets.delete_local(dataset_id)


@app.post("/datasets/upload-to-db")
def upload_to_db(
    dataset_id: str = Query(..., description="Dataset ID to upload"),
    batch_size: int = Query(1000, description="Batch size for upload"),
    service: IndexingService = Depends(get_service),
):
    """Load documents from a local ir_datasets dataset into PostgreSQL."""
    return service.datasets.upload(dataset_id, batch_size=batch_size)


@app.get("/datasets/db-info")
def db_info(service: IndexingService = Depends(get_service)):
    """Row counts of the documents database."""
    return service.datasets.database_info()


@app.get("/datasets/metadata")
def list_datasets_metadata(service: IndexingService = Depends(get_service)):
    """Every dataset with stored metadata."""
    datasets = service.datasets.metadata()
    return {"datasets": datasets, "count": len(datasets)}


@app.delete("/datasets/db")
def delete_dataset_from_db_endpoint(
    dataset_id: str = Query(..., description="Dataset ID to delete from DB"),
    service: IndexingService = Depends(get_service),
):
    """Delete all documents of one dataset from PostgreSQL."""
    return service.datasets.delete_from_database(dataset_id)


@app.post("/datasets/fix-corrupted")
def fix_corrupted_dataset(
    dataset_id: str = Query(..., description="Dataset ID to fix"),
    service: IndexingService = Depends(get_service),
):
    """Delete a corrupted local dataset and download it again."""
    return service.datasets.fix_corrupted(dataset_id)


# ----------------------------------------------------------------------------------------------- index builds
@app.get("/index/stats")
def index_stats(
    dataset_name: str = Query(..., description="Dataset name"),
    doc_count: Optional[int] = Query(None, description="Total number of documents to report (optional)"),
    service: IndexingService = Depends(get_service),
):
    return service.stats.all_indices(dataset_name, doc_count)


@app.post("/index/tfidf/build")
def build_tfidf(
    dataset_name: str = Query(..., description="Dataset name to build index for"),
    force_rebuild: bool = Query(False, description="Force rebuild the index"),
    body: Optional[dict] = Body(None),
    service: IndexingService = Depends(get_service),
):
    return service.builds.tfidf(dataset_name, force_rebuild, _preprocessing_config(body))


@app.get("/index/tfidf/status")
def tfidf_status(
    dataset_name: str = Query(..., description="Dataset name"),
    service: IndexingService = Depends(get_service),
):
    return service.stats.tfidf(dataset_name)


@app.post("/index/embedding/build")
def build_embedding(
    dataset_name: str = Query(..., description="Dataset name to build index for"),
    force_rebuild: bool = Query(False, description="Force rebuild the index"),
    body: Optional[dict] = Body(None),
    service: IndexingService = Depends(get_service),
):
    return service.builds.embedding(dataset_name, force_rebuild, _preprocessing_config(body))


@app.get("/index/embedding/status")
def embedding_status(
    dataset_name: str = Query(..., description="Dataset name"),
    service: IndexingService = Depends(get_service),
):
    return service.stats.embedding(dataset_name)


@app.post("/index/bm25/build")
def build_bm25(
    dataset_name: str = Query(..., description="Dataset name to build index for"),
    force_rebuild: bool = Query(False, description="Force rebuild the index"),
    body: Optional[dict] = Body(None),
    service: IndexingService = Depends(get_service),
):
    return service.builds.bm25(dataset_name, force_rebuild, _preprocessing_config(body))


@app.get("/index/bm25/status")
def bm25_status(
    dataset_name: str = Query(..., description="Dataset name"),
    service: IndexingService = Depends(get_service),
):
    return service.stats.bm25(dataset_name)


@app.post("/index/build-all")
def build_all(
    dataset_name: str = Query(..., description="Dataset name to build indices for"),
    force_rebuild: bool = Query(False, description="Force rebuild all indices"),
    body: Optional[dict] = Body(None),
    service: IndexingService = Depends(get_service),
):
    """Build TF-IDF, BM25 and embedding indices with one shared preprocessing pass."""
    return service.builds.all(dataset_name, force_rebuild, _preprocessing_config(body))


# ----------------------------------------------------------------------------------------------- inverted index
@app.get("/index/inverted/stats")
def inverted_stats(
    dataset_name: str = Query(..., description="Dataset name"),
    service: IndexingService = Depends(get_service),
):
    """Comprehensive inverted index statistics."""
    return service.stats.inverted_index(dataset_name)


@app.get("/index/inverted/term/{term}")
def term_info(
    term: str,
    dataset_name: str = Query(..., description="Dataset name"),
    service: IndexingService = Depends(get_service),
):
    """Detailed information about one term."""
    return service.stats.term(dataset_name, term)


@app.get("/index/inverted/search")
def search_vocabulary(
    prefix: str = Query(..., description="Prefix to search for"),
    limit: int = Query(50, description="Maximum number of results"),
    dataset_name: str = Query(..., description="Dataset name"),
    service: IndexingService = Depends(get_service),
):
    """Terms starting with the given prefix."""
    return service.stats.search_terms(dataset_name, prefix, limit)


@app.get("/index/inverted/sample")
def vocabulary_sample(
    sample_size: int = Query(100, description="Sample size"),
    dataset_name: str = Query(..., description="Dataset name"),
    service: IndexingService = Depends(get_service),
):
    """A random sample of vocabulary terms."""
    return service.stats.vocabulary_sample(dataset_name, sample_size)


# ----------------------------------------------------------------------------------------------- deletion
@app.delete("/index/delete-all")
def delete_all_indices_endpoint(
    dataset_name: str = Query(..., description="Dataset name"),
    service: IndexingService = Depends(get_service),
):
    """Delete every index file of a dataset."""
    return service.indices.delete_all(dataset_name)


@app.delete("/index/{index_type}/delete")
def delete_index(
    index_type: str,
    dataset_name: str = Query(..., description="Dataset name"),
    service: IndexingService = Depends(get_service),
):
    """Delete one index type of a dataset."""
    return service.indices.delete(index_type, dataset_name)


# --- shared HTTP protections -------------------------------------------------
from common.security import install_cors  # noqa: E402

install_cors(app)  # installed last, so the policy also covers throttled responses
