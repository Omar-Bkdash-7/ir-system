"""Response bodies of the reporting and maintenance endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetsResponse(BaseModel):
    datasets: List[str] = Field(description="Dataset names offered for evaluation")


class StatusResponse(BaseModel):
    service: str
    status: str
    supported_models: List[str]
    supported_metrics: List[str]


class ModelMetrics(BaseModel):
    """One model's latest run, as `/metrics` reports it."""

    map: float
    ndcg_at_10: float
    recall: float
    precision_at_10: float
    num_queries: int


class HistoryResponse(BaseModel):
    history: List[Dict[str, Any]] = Field(description="Stored runs, newest first")


class CompareResponse(BaseModel):
    dataset: str
    top_k: int
    models: List[Dict[str, Any]]


class DeleteResultResponse(BaseModel):
    success: bool
    message: str


class DeleteResultsResponse(BaseModel):
    success: bool
    message: str
    count: int
    filters: Dict[str, Optional[str]]


class ClearResultsResponse(BaseModel):
    success: bool
    message: str
    count: int
