from pydantic import BaseModel
from typing import List

class RankingResponse(BaseModel):
    ranked_results: List[dict]
    total: int
