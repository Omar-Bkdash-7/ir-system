"""Request bodies of the API gateway, and their validation.

These are the contracts the frontend sends; the shapes are unchanged, they simply
live beside the routes instead of inside them.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class BasicSearchRequest(BaseModel):
    query: str = Field(description="Search query text")
    model: str = Field(default="tfidf", description="Model: tfidf, embedding, or bm25")
    top_k: int = Field(default=50, ge=1, le=1000, description="Number of results")
    bm25_k1: Optional[float] = Field(default=1.5, ge=0.0, le=3.0)
    bm25_b: Optional[float] = Field(default=0.75, ge=0.0, le=1.0)
    enable_refinement: bool = Field(default=False, description="Enable query refinement")
    query_history: List[str] = Field(default_factory=list, description="Query history for refinement")
    dataset: Optional[str] = Field(default=None, description="Dataset name (e.g., 'cranfield', 'vaswani')")


class HybridSerialSearchRequest(BaseModel):
    query: str = Field(description="Search query text")
    first_model: str = Field(description="First stage model: tfidf, embedding, or bm25")
    second_model: str = Field(description="Second stage model: tfidf, embedding, or bm25")
    first_stage_k: int = Field(default=1000, ge=10, le=5000, description="Number of candidates from first stage")
    final_top_k: int = Field(default=50, ge=1, le=1000, description="Final number of results")
    bm25_k1: Optional[float] = Field(default=1.5, ge=0.0, le=3.0)
    bm25_b: Optional[float] = Field(default=0.75, ge=0.0, le=1.0)
    enable_refinement: bool = Field(default=False, description="Enable query refinement")
    query_history: List[str] = Field(default_factory=list, description="Query history for refinement")
    dataset: Optional[str] = Field(default=None, description="Dataset name (e.g., 'cranfield', 'vaswani')")


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
    query_history: List[str] = Field(default_factory=list, description="Query history for refinement")
    dataset: Optional[str] = Field(default=None, description="Dataset name (e.g., 'cranfield', 'vaswani')")

    @model_validator(mode="after")
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
    history: List[str] = Field(default_factory=list, description="Query history")
    enabled: bool = Field(default=True, description="Enable refinement")
    session_id: str = Field(default="default", description="Session identifier")
    is_evaluation: bool = Field(default=False, description="Evaluation mode (deterministic spelling only)")
    dataset: Optional[str] = Field(default=None, description="Dataset whose vocabulary protects domain terms")
