from pydantic import BaseModel
from typing import List

class RetrievalRequest(BaseModel):
    query_tokens: List[str]
    top_k: int = 50
    include_text: bool = True

class RetrievalResponse(BaseModel):
    results: List[dict]
    total: int
