from typing import Any, Dict, List

from pydantic import BaseModel, Field

MAX_RESULTS = 100_000
MAX_TOP_K = 10_000


class RankingRequest(BaseModel):
    results: List[Dict[str, Any]] = Field(
        max_length=MAX_RESULTS,
        description="Results to order. Each item needs a numeric 'score' (higher is better); "
                    "every other field (doc_id, text, metadata, ...) is returned unchanged",
    )
    method: str = Field(
        default="cosine",
        description="'cosine' (the original name) or 'score': both order the results by their 'score', best first",
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=MAX_TOP_K,
        description="How many results to return",
    )
