from fastapi import FastAPI, HTTPException
from services.api_gateway.dataset_client import get_dataset_info

app = FastAPI(title="API Gateway")

@app.get("/health")
def health():
    return {"status": "ok", "service": "api_gateway"}

@app.get("/datasets/info")
def datasets_info():
    try:
        return get_dataset_info()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Dataset metadata not found. Please run indexing service validation first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metadata: {str(e)}")
