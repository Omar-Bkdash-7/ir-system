from fastapi import FastAPI, HTTPException
from services.indexing.dataset_service import (
    validate_and_store_metadata,
    get_metadata,
    DatasetValidationError,
    ForbiddenDatasetError
)

app = FastAPI(title="Indexing Service")

@app.get("/health")
def health():
    return {"status": "ok", "service": "indexing"}

@app.post("/validate")
def validate_dataset(dataset_id: str = "beir/quora"):
    try:
        metadata = validate_and_store_metadata(dataset_id)
        return metadata
    except ForbiddenDatasetError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except DatasetValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/index/stats")
def index_stats():
    try:
        metadata = get_metadata()
        return {
            "vocab_size": None,
            "total_docs": metadata["doc_count"],
            "avg_doc_length": None
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metadata not found. Run /validate first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
