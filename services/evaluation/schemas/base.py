from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class EvaluationMetrics(BaseModel):
    """Metrics for evaluation results"""
    map: float = Field(description="Mean Average Precision")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain")
    recall: float = Field(description="Recall at k")
    precision: float = Field(description="Precision at k")
    num_queries: int = Field(description="Number of queries evaluated")


class QueryEvaluationResult(BaseModel):
    """Result for a single query evaluation"""
    query_id: str = Field(description="Query identifier")
    query_text: str = Field(description="Query text")
    num_retrieved: int = Field(description="Number of documents retrieved")
    num_relevant: int = Field(description="Number of relevant documents")
    num_relevant_retrieved: int = Field(description="Number of relevant documents retrieved")
    ap: float = Field(description="Average Precision")
    ndcg: float = Field(description="NDCG score")
    recall: float = Field(description="Recall score")
    precision: float = Field(description="Precision score")


class EvaluationResponse(BaseModel):
    """Response for evaluation endpoint"""
    model: str = Field(description="Model evaluated")
    top_k: int = Field(description="Top K used for evaluation")
    metrics: EvaluationMetrics = Field(description="Overall metrics")
    query_results: Optional[List[QueryEvaluationResult]] = Field(
        default=None,
        description="Per-query results (optional)"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Queries that could not be scored (partial results), if any"
    )
