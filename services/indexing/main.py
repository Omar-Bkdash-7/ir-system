from fastapi import FastAPI
from services.indexing.dataset_service import validate_and_store_metadata
from services.indexing.indexing_service import (
    get_index_stats,
    build_tfidf_index,
    get_tfidf_status
)
from common.exception_handlers import register_exception_handlers
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

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
