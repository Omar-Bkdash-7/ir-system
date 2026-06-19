from fastapi import FastAPI, HTTPException
from services.api_gateway.clients import get_dataset_info, search_hybrid_serial
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
from pydantic import BaseModel, Field
from typing import List, Optional

logger = setup_logging("api_gateway")

app = FastAPI(title="API Gateway")
register_exception_handlers(app)


class HybridSerialSearchRequest(BaseModel):
    query: str = Field(description="Search query text")
    first_model: str = Field(description="First stage model: tfidf, embedding, or bm25")
    second_model: str = Field(description="Second stage model: tfidf, embedding, or bm25")
    first_stage_k: int = Field(default=1000, ge=10, le=5000, description="Number of candidates from first stage")
    final_top_k: int = Field(default=50, ge=1, le=1000, description="Final number of results")
    bm25_k1: Optional[float] = Field(default=1.5, ge=0.0, le=3.0)
    bm25_b: Optional[float] = Field(default=0.75, ge=0.0, le=1.0)


@app.get("/health")
def health():
    return {"status": "ok", "service": "api_gateway"}


@app.get("/datasets/info")
def datasets_info():
    return get_dataset_info()


@app.post("/search/hybrid/serial")
def search_hybrid_serial_endpoint(req: HybridSerialSearchRequest):
    try:
        results = search_hybrid_serial(
            query=req.query,
            first_model=req.first_model,
            second_model=req.second_model,
            first_stage_k=req.first_stage_k,
            final_top_k=req.final_top_k,
            bm25_k1=req.bm25_k1,
            bm25_b=req.bm25_b
        )
        return results
    except Exception as e:
        logger.error(f"Hybrid serial search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
