"""Framework-free core of the evaluation service: clients, query preparation, the run loop, reports."""
from services.evaluation.core.clients import JsonClient, UpstreamServiceError
from services.evaluation.core.dataset import DatasetSplit
from services.evaluation.core.reports import latest_metrics_by_model, per_query_analysis, refinement_comparison
from services.evaluation.core.runner import RunResult, run_over_queries
from services.evaluation.core.query import PreparedQuery, QueryPipeline
from services.evaluation.core.service import EvaluationService

__all__ = [
    "DatasetSplit",
    "EvaluationService",
    "JsonClient",
    "PreparedQuery",
    "QueryPipeline",
    "RunResult",
    "UpstreamServiceError",
    "latest_metrics_by_model",
    "per_query_analysis",
    "refinement_comparison",
    "run_over_queries",
]
