from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RankingResponse(BaseModel):
    ranked_results: List[Dict[str, Any]] = Field(description="The ordered results, exactly as they were sent")
    total: int = Field(description="Number of results returned")
    took_ms: Optional[float] = Field(default=None, description="Server-side processing time in milliseconds")
