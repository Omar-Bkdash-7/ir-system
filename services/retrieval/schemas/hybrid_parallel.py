from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict


class HybridParallelRequest(BaseModel):
    query_tokens: List[str] = Field(
        default_factory=list,
        description="Preprocessed query tokens"
    )
    query_text: str = Field(
        description="Original query text"
    )
    models: List[str] = Field(
        description="List of models to run in parallel: ['tfidf', 'embedding', 'bm25']",
        min_length=2,
        max_length=3
    )
    fusion_method: str = Field(
        default="rrf",
        description="Fusion method: 'rrf', 'combsum', 'combmnz', 'weighted'"
    )
    model_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights per model for weighted fusion (must sum to 1.0)"
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Final number of results after fusion"
    )
    candidates_per_model: int = Field(
        default=1000,
        ge=10,
        le=5000,
        description="Number of candidates to retrieve per model"
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=100,
        description="RRF constant k parameter"
    )
    include_text: bool = Field(
        default=True,
        description="Whether to include document text in results"
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
    dataset: Optional[str] = Field(
        default=None,
        description="Dataset name (e.g., 'cranfield', 'vaswani')"
    )

    @field_validator("models")
    @classmethod
    def validate_models(cls, v: List[str]) -> List[str]:
        allowed = ["tfidf", "embedding", "bm25"]
        normalized = [m.lower() for m in v]
        
        for model in normalized:
            if model not in allowed:
                raise ValueError(f"Model must be one of {allowed}, got '{model}'")
        
        if len(normalized) != len(set(normalized)):
            raise ValueError("Models list contains duplicates")
        
        if len(normalized) < 2:
            raise ValueError("Parallel mode requires at least 2 models")
        
        return normalized
    
    @field_validator("fusion_method")
    @classmethod
    def validate_fusion_method(cls, v: str) -> str:
        allowed = ["rrf", "combsum", "combmnz", "weighted"]
        if v.lower() not in allowed:
            raise ValueError(f"Fusion method must be one of {allowed}, got '{v}'")
        return v.lower()
    
    @model_validator(mode='after')
    def validate_weights(self):
        if self.fusion_method == "weighted":
            if not self.model_weights:
                raise ValueError("model_weights is required when fusion_method='weighted'")
            
            for model in self.models:
                if model not in self.model_weights:
                    raise ValueError(f"Weight for model '{model}' is missing")
            
            total_weight = sum(self.model_weights.values())
            if not (0.99 <= total_weight <= 1.01):
                raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        return self
