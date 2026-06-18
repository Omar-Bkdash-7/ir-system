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
        description="Retrieval model: 'tfidf' or 'embedding'"
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

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed = ["tfidf", "embedding"]
        if v.lower() not in allowed:
            raise ValueError(f"Model must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("query_tokens")
    @classmethod
    def validate_query_tokens(cls, v: List[str]) -> List[str]:
        if v and not all(isinstance(token, str) for token in v):
            raise ValueError("All query_tokens must be strings")
        return v


class RetrievalResponse(BaseModel):
    results: List[dict] = Field(
        description="Retrieved documents with scores"
    )
    total: int = Field(
        description="Total number of results returned"
    )
    model: str = Field(
        default="tfidf",
        description="Model used for retrieval"
    )
