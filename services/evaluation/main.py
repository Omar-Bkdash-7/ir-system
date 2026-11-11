"""Evaluation service: the HTTP layer. It only validates requests and calls core.service."""
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request

from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
from services.evaluation.core.names import METRIC_NAMES, SUPPORTED_MODELS
from services.evaluation.core.service import EvaluationService
from services.evaluation.schemas import EvaluationRequest, EvaluationResponse
from services.evaluation.schemas.responses import (
    ClearResultsResponse,
    CompareResponse,
    DatasetsResponse,
    DeleteResultResponse,
    DeleteResultsResponse,
    HistoryResponse,
    ModelMetrics,
    StatusResponse,
)

logger = setup_logging("evaluation")


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = EvaluationService.create()
    app.state.service = service
    logger.info("Evaluation service started (settings from configs/config.yaml and EVALUATION_* variables)")
    try:
        yield
    finally:
        service.close()
        logger.info("Evaluation service stopped")


app = FastAPI(title="Evaluation Service", lifespan=lifespan)
register_exception_handlers(app)


def get_service(request: Request) -> EvaluationService:
    service = getattr(request.app.state, "service", None)
    if service is None:  # e.g. a TestClient used without running the lifespan
        service = request.app.state.service = EvaluationService.create()
    return service


# ----------------------------------------------------------------------------------------------- health & info
@app.get("/health")
def health():
    return {"status": "ok", "service": "evaluation"}


@app.get("/status", response_model=StatusResponse)
def get_status():
    return {
        "service": "evaluation",
        "status": "ready",
        "supported_models": list(SUPPORTED_MODELS),
        "supported_metrics": list(METRIC_NAMES),
    }


@app.get("/datasets", response_model=DatasetsResponse)
def get_datasets(service: EvaluationService = Depends(get_service)):
    return {"datasets": service.get_available_datasets()}


# ----------------------------------------------------------------------------------------------- evaluation
@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest, service: EvaluationService = Depends(get_service)):
    """
    Evaluate a retrieval model.

    Calculates standard IR metrics: MAP, nDCG, Recall@k, Precision@k.
    """
    return service.evaluate(request)


@app.post("/run", response_model=EvaluationResponse)
def run_evaluation(
    model: str,
    dataset: str = "beir/quora/test",
    top_k: int = 100,
    query_limit: Optional[int] = None,
    service: EvaluationService = Depends(get_service),
):
    """Simplified endpoint for frontend integration: the same run, with query parameters."""
    return service.evaluate(
        EvaluationRequest(
            dataset_name=dataset,
            model=model,
            top_k=top_k,
            query_limit=query_limit,
            include_per_query=False,
        )
    )


# ----------------------------------------------------------------------------------------------- reporting
@app.get("/history", response_model=HistoryResponse)
def get_history(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    limit: int = 10,
    service: EvaluationService = Depends(get_service),
):
    return {"history": service.get_evaluation_history(model=model, dataset=dataset, limit=limit)}


@app.get("/compare", response_model=CompareResponse)
def compare_models(dataset: str, top_k: int = 100, service: EvaluationService = Depends(get_service)):
    return {
        "dataset": dataset,
        "top_k": top_k,
        "models": service.compare_models(dataset=dataset, top_k=top_k),
    }


@app.get("/metrics", response_model=Dict[str, ModelMetrics])
def get_metrics(service: EvaluationService = Depends(get_service)):
    """Latest evaluation metrics for every model, in the shape the frontend expects."""
    return service.get_latest_metrics()


@app.get("/metrics/per-query")
def get_per_query_analysis(service: EvaluationService = Depends(get_service)):
    return service.get_per_query_analysis()


@app.get("/metrics/refinement-comparison")
def get_refinement_comparison(service: EvaluationService = Depends(get_service)):
    return service.get_refinement_comparison()


# ----------------------------------------------------------------------------------------------- maintenance
@app.delete("/result/{result_id}", response_model=DeleteResultResponse)
def delete_result(result_id: int, service: EvaluationService = Depends(get_service)):
    if not service.delete_result(result_id):
        raise HTTPException(status_code=404, detail=f"Evaluation result {result_id} not found")
    return {"success": True, "message": f"Evaluation result {result_id} deleted successfully"}


@app.delete("/results", response_model=DeleteResultsResponse)
def delete_results(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    before_date: Optional[str] = None,
    service: EvaluationService = Depends(get_service),
):
    """
    Delete evaluation results by filters. At least one filter is required.

    Query parameters:
    - model: filter by model name
    - dataset: filter by dataset name
    - before_date: delete results before this date (ISO format: YYYY-MM-DD)
    """
    if not any((model, dataset, before_date)):
        raise HTTPException(
            status_code=400,
            detail="At least one filter parameter (model, dataset, or before_date) is required",
        )

    count = service.delete_results(model=model, dataset=dataset, before_date=before_date)
    return {
        "success": True,
        "message": f"Deleted {count} evaluation results",
        "count": count,
        "filters": {"model": model, "dataset": dataset, "before_date": before_date},
    }


@app.delete("/results/clear-all", response_model=ClearResultsResponse)
def clear_all_results(service: EvaluationService = Depends(get_service)):
    """Delete every stored evaluation result. This cannot be undone."""
    count = service.clear_results()
    return {"success": True, "message": "All evaluation results cleared", "count": count}
