from pydantic import BaseModel, Field
from typing import List


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
