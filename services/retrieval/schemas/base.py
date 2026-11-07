from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SearchHit(BaseModel):
    # ids that were stored as numbers are returned as text, so a numeric id can never fail response validation
    model_config = ConfigDict(coerce_numbers_to_str=True)

    doc_id: str = Field(description="Document id")
    score: float = Field(description="Relevance score (its scale depends on the model)")
    text: Optional[str] = Field(default=None, description="Document text (present when include_text is true)")


class RetrievalResponse(BaseModel):
    results: List[SearchHit] = Field(
        description="Retrieved documents with scores"
    )
    total: int = Field(
        description="Total number of results returned"
    )
    model: str = Field(
        default="tfidf",
        description="Model used for retrieval"
    )
    took_ms: Optional[float] = Field(
        default=None,
        description="Server-side processing time in milliseconds"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Only present when part of the work degraded, e.g. one model of a hybrid search failed"
    )
