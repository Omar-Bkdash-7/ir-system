from fastapi import FastAPI, HTTPException
from services.api_gateway.clients import (
    get_dataset_info,
    search_hybrid_serial,
    search_hybrid_parallel,
    search_basic,
    refine_query
)
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict

logger = setup_logging("api_gateway")

app = FastAPI(title="API Gateway")
register_exception_handlers(app)


class BasicSearchRequest(BaseModel):
    query: str = Field(description="Search query text")
    model: str = Field(default="tfidf", description="Model: tfidf, embedding, or bm25")
    top_k: int = Field(default=50, ge=1, le=1000, description="Number of results")
    bm25_k1: Optional[float] = Field(default=1.5, ge=0.0, le=3.0)
    bm25_b: Optional[float] = Field(default=0.75, ge=0.0, le=1.0)
    enable_refinement: bool = Field(default=False, description="Enable query refinement")
    query_history: List[str] = Field(default=[], description="Query history for refinement")


class HybridSerialSearchRequest(BaseModel):
    query: str = Field(description="Search query text")
    first_model: str = Field(description="First stage model: tfidf, embedding, or bm25")
    second_model: str = Field(description="Second stage model: tfidf, embedding, or bm25")
    first_stage_k: int = Field(default=1000, ge=10, le=5000, description="Number of candidates from first stage")
    final_top_k: int = Field(default=50, ge=1, le=1000, description="Final number of results")
    bm25_k1: Optional[float] = Field(default=1.5, ge=0.0, le=3.0)
    bm25_b: Optional[float] = Field(default=0.75, ge=0.0, le=1.0)
    enable_refinement: bool = Field(default=False, description="Enable query refinement")
    query_history: List[str] = Field(default=[], description="Query history for refinement")


class HybridParallelSearchRequest(BaseModel):
    query: str = Field(description="Search query text")
    models: List[str] = Field(description="Models to run in parallel", min_length=2, max_length=3)
    fusion_method: str = Field(default="rrf", description="Fusion method: rrf, combsum, combmnz, weighted")
    model_weights: Optional[Dict[str, float]] = Field(default=None, description="Weights for weighted fusion")
    top_k: int = Field(default=50, ge=1, le=1000, description="Final number of results")
    candidates_per_model: int = Field(default=1000, ge=10, le=5000, description="Candidates per model")
    rrf_k: int = Field(default=60, ge=1, le=100, description="RRF constant")
    bm25_k1: Optional[float] = Field(default=1.5, ge=0.0, le=3.0)
    bm25_b: Optional[float] = Field(default=0.75, ge=0.0, le=1.0)
    enable_refinement: bool = Field(default=False, description="Enable query refinement")
    query_history: List[str] = Field(default=[], description="Query history for refinement")
    
    @model_validator(mode='after')
    def validate_weights(self):
        if self.fusion_method == "weighted":
            if not self.model_weights:
                raise ValueError("model_weights required when fusion_method='weighted'")
            for model in self.models:
                if model not in self.model_weights:
                    raise ValueError(f"Weight for model '{model}' is missing")
            total = sum(self.model_weights.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Weights must sum to 1.0, got {total}")
        return self


class QueryRefineRequest(BaseModel):
    query: str = Field(description="Query to refine")
    history: List[str] = Field(default=[], description="Query history")
    enabled: bool = Field(default=True, description="Enable refinement")


@app.get("/health")
def health():
    return {"status": "ok", "service": "api_gateway"}


@app.get("/datasets/info")
def datasets_info():
    return get_dataset_info()


@app.post("/query/refine")
def query_refine_endpoint(req: QueryRefineRequest):
    try:
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        result = refine_query(req.query, req.history, req.enabled)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query refinement failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_endpoint(req: BasicSearchRequest):
    try:
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        query_to_use = req.query
        refinement_info = None
        
        if req.enable_refinement:
            refinement_result = refine_query(req.query, req.query_history, True)
            query_to_use = refinement_result.get("refined_query", req.query)
            refinement_info = refinement_result
        
        results = search_basic(
            query=query_to_use,
            model=req.model,
            top_k=req.top_k,
            bm25_k1=req.bm25_k1,
            bm25_b=req.bm25_b
        )
        
        if refinement_info:
            results["refinement"] = refinement_info
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Basic search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid/serial")
def search_hybrid_serial_endpoint(req: HybridSerialSearchRequest):
    try:
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        query_to_use = req.query
        refinement_info = None
        
        if req.enable_refinement:
            refinement_result = refine_query(req.query, req.query_history, True)
            query_to_use = refinement_result.get("refined_query", req.query)
            refinement_info = refinement_result
        
        results = search_hybrid_serial(
            query=query_to_use,
            first_model=req.first_model,
            second_model=req.second_model,
            first_stage_k=req.first_stage_k,
            final_top_k=req.final_top_k,
            bm25_k1=req.bm25_k1,
            bm25_b=req.bm25_b
        )
        
        if refinement_info:
            results["refinement"] = refinement_info
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid serial search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid/parallel")
def search_hybrid_parallel_endpoint(req: HybridParallelSearchRequest):
    try:
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        query_to_use = req.query
        refinement_info = None
        
        if req.enable_refinement:
            refinement_result = refine_query(req.query, req.query_history, True)
            query_to_use = refinement_result.get("refined_query", req.query)
            refinement_info = refinement_result
        
        results = search_hybrid_parallel(
            query=query_to_use,
            models=req.models,
            fusion_method=req.fusion_method,
            model_weights=req.model_weights,
            top_k=req.top_k,
            candidates_per_model=req.candidates_per_model,
            rrf_k=req.rrf_k,
            bm25_k1=req.bm25_k1,
            bm25_b=req.bm25_b
        )
        
        if refinement_info:
            results["refinement"] = refinement_info
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid parallel search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
