from pydantic import BaseModel
from typing import List

class RankingRequest(BaseModel):
    results: List[dict]
    method: str = "cosine"
    top_k: int = 50
