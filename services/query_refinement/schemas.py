from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RefineRequest(BaseModel):
    query: str = Field(..., description="Query to refine", min_length=1)
    history: List[str] = Field(default_factory=list, description="Previous queries for context")
    enabled: bool = Field(default=True, description="Enable refinement")
    session_id: str = Field(default="default", description="Session identifier")
    is_evaluation: bool = Field(default=False, description="Evaluation mode (deterministic spelling only)")
    dataset: Optional[str] = Field(
        default=None, description="Dataset being searched; its vocabulary protects domain terms from correction"
    )


class RefineResponse(BaseModel):
    original_query: str = Field(..., description="Original query")
    refined_query: str = Field(..., description="Refined query")
    corrections: Dict[str, str] = Field(default_factory=dict, description="Spelling corrections applied")
    expansions: Dict[str, List[str]] = Field(default_factory=dict, description="WordNet synonyms added per word")
    history_boosts: Dict[str, float] = Field(default_factory=dict, description="History terms and their boost weight")
    history_terms: List[str] = Field(default_factory=list, description="Terms taken from the session history")
    history_similarity: float = Field(default=0.0, description="Similarity between the query and the session history")
    confidence: float = Field(default=0.0, description="Refinement confidence score")
    changes_applied: List[str] = Field(default_factory=list, description="Stages that changed the query")
    semantic_drift: float = Field(default=0.0, description="Measured drift from the original query")
    applied: bool = Field(..., description="Whether the refined query differs from the original")
    evaluation_mode: bool = Field(default=False, description="Deterministic evaluation mode was used")
    drift_rejected: bool = Field(default=False, description="Expansion was reverted because the drift was too high")
