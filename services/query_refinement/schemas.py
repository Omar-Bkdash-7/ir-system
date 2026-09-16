from pydantic import BaseModel, Field
from typing import List, Dict

class RefineRequest(BaseModel):
    query: str = Field(..., description="Query to refine", min_length=1)
    history: List[str] = Field(default=[], description="Previous queries for context")
    enabled: bool = Field(default=True, description="Enable refinement")
    session_id: str = Field(default="default", description="Session identifier")
    is_evaluation: bool = Field(default=False, description="Evaluation mode (spelling only)")

class RefineResponse(BaseModel):
    original_query: str = Field(..., description="Original query")
    refined_query: str = Field(..., description="Refined query")
    corrections: Dict[str, str] = Field(default={}, description="Spelling corrections")
    expansions: Dict[str, List[str]] = Field(default={}, description="Synonym expansions")
    history_boosts: Dict[str, float] = Field(default={}, description="History term boosts")
    confidence: float = Field(default=0.0, description="Refinement confidence score")
    changes_applied: List[str] = Field(default=[], description="Types of changes applied")
    semantic_drift: float = Field(default=0.0, description="Semantic drift from original")
    applied: bool = Field(..., description="Whether refinement was applied")

