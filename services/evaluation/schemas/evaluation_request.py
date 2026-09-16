from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class EvaluationRequest(BaseModel):
    """Request for evaluation endpoint"""
    
    dataset_name: str = Field(
        default="beir/quora/test",
        description="Dataset name for evaluation (must have qrels)"
    )
    
    model: str = Field(
        description="Model to evaluate: 'tfidf', 'embedding', 'bm25', 'hybrid_serial', 'hybrid_parallel'"
    )
    
    top_k: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of documents to evaluate"
    )
    
    query_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Limit number of queries to evaluate (for testing)"
    )
    
    include_per_query: bool = Field(
        default=False,
        description="Include per-query results in response"
    )
    
    bm25_k1: Optional[float] = Field(
        default=1.5,
        ge=0.0,
        le=3.0,
        description="BM25 k1 parameter"
    )
    
    bm25_b: Optional[float] = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 b parameter"
    )
    
    hybrid_models: Optional[List[str]] = Field(
        default=None,
        description="Models for hybrid evaluation"
    )
    
    fusion_method: Optional[str] = Field(
        default="rrf",
        description="Fusion method for hybrid parallel: 'rrf', 'weighted'"
    )
    
    enable_refinement: bool = Field(
        default=False,
        description="Enable query refinement during evaluation"
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed = ["tfidf", "embedding", "bm25", "hybrid_serial", "hybrid_parallel"]
        if v.lower() not in allowed:
            raise ValueError(f"Model must be one of {allowed}, got '{v}'")
        return v.lower()
