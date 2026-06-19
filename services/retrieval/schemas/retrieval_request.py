from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class RetrievalRequest(BaseModel):
    query_tokens: List[str] = Field(
        default_factory=list,
        description="Preprocessed query tokens (required for TF-IDF)"
    )
    query_text: Optional[str] = Field(
        default="",
        description="Original query text (required for embedding)"
    )
    model: str = Field(
        default="tfidf",
        description="Retrieval model: 'tfidf', 'embedding', or 'bm25'"
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Number of documents to retrieve"
    )
    include_text: bool = Field(
        default=True,
        description="Whether to include document text in results"
    )
    bm25_k1: Optional[float] = Field(
        default=1.5,
        ge=0.0,
        le=3.0,
        description="BM25 k1 parameter (term saturation)"
    )
    bm25_b: Optional[float] = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 b parameter (length normalization)"
    )

    @field_validator("model")
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
