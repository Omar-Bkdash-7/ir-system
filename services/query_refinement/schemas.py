from pydantic import BaseModel, Field
from typing import List, Dict

class RefineRequest(BaseModel):
    query: str = Field(..., description="Query to refine", min_length=1)
    history: List[str] = Field(default=[], description="Previous queries for context")
    enabled: bool = Field(default=True, description="Enable refinement")
    session_id: str = Field(default="default", description="Session identifier")

class RefineResponse(BaseModel):
    original_query: str = Field(..., description="Original query")
    refined_query: str = Field(..., description="Refined query")
    corrections: Dict[str, str] = Field(default={}, description="Spelling corrections")
    synonyms_added: Dict[str, List[str]] = Field(default={}, description="Synonyms added")
    history_terms_added: List[str] = Field(default=[], description="Terms added from history")
    history_similarity: float = Field(default=0.0, description="Similarity with history")
    boosted_terms: List[str] = Field(default=[], description="Terms boosted from history")
    applied: bool = Field(..., description="Whether refinement was applied")
