from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class HybridSerialRequest(BaseModel):
    query_tokens: List[str] = Field(
        default_factory=list,
        description="Preprocessed query tokens"
    )
    query_text: str = Field(
        description="Original query text"
    )
    first_model: str = Field(
        description="First stage model: 'tfidf', 'embedding', or 'bm25'"
    )
    second_model: str = Field(
        description="Second stage model: 'tfidf', 'embedding', or 'bm25'"
    )
    first_stage_k: int = Field(
        default=1000,
        ge=10,
        le=5000,
        description="Number of candidates from first stage"
    )
    final_top_k: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Final number of results after re-ranking"
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

    @field_validator("first_model", "second_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed = ["tfidf", "embedding", "bm25"]
        if v.lower() not in allowed:
            raise ValueError(f"Model must be one of {allowed}, got '{v}'")
        return v.lower()
    
    @field_validator("query_tokens")
    @classmethod
    def validate_query_tokens(cls, v: List[str]) -> List[str]:
        if v and not all(isinstance(token, str) for token in v):
            raise ValueError("All query_tokens must be strings")
        return v
