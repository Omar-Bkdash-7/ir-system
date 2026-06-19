from fastapi import FastAPI
from services.indexing.dataset_service import validate_and_store_metadata
from services.indexing.indexing_service import (
    get_index_stats,
    build_tfidf_index,
    get_tfidf_status,
    build_embedding_index,
    get_embedding_status,
    build_bm25_index,
    get_bm25_status
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
def validate_dataset(dataset_id: str = "beir/quora"):
    return validate_and_store_metadata(dataset_id)


@app.get("/index/stats")
def index_stats():
    return get_index_stats()


@app.post("/index/tfidf/build")
def build_tfidf(force_rebuild: bool = False):
    return build_tfidf_index(force_rebuild)


@app.get("/index/tfidf/status")
def tfidf_status():
    return get_tfidf_status()


@app.post("/index/embedding/build")
def build_embedding(force_rebuild: bool = False):
    return build_embedding_index(force_rebuild)


@app.get("/index/embedding/status")
def embedding_status():
    return get_embedding_status()


@app.post("/index/bm25/build")
def build_bm25(force_rebuild: bool = False):
    return build_bm25_index(force_rebuild)


@app.get("/index/bm25/status")
def bm25_status():
    return get_bm25_status()
