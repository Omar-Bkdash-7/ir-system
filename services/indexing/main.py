from fastapi import FastAPI, HTTPException
from services.indexing.dataset_service import (
    validate_and_store_metadata,
    DatasetValidationError,
    ForbiddenDatasetError
)
from services.indexing.indexing_service import (
    get_index_stats,
    build_tfidf_index,
    get_tfidf_status
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Indexing Service")

@app.get("/health")
def health():
    return {"status": "ok", "service": "indexing"}

@app.post("/validate")
def validate_dataset(dataset_id: str = "beir/quora"):
    try:
        return validate_and_store_metadata(dataset_id)
    except ForbiddenDatasetError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except DatasetValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/index/stats")
def index_stats():
    try:
        return get_index_stats()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/tfidf/build")
def build_tfidf(force_rebuild: bool = False):
    try:
        return build_tfidf_index(force_rebuild)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building TF-IDF index: {str(e)}")

@app.get("/index/tfidf/status")
def tfidf_status():
    try:
        return get_tfidf_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
