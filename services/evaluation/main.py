from fastapi import FastAPI, HTTPException
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
from services.evaluation.evaluation_service import EvaluationService
from services.evaluation.schemas import (
    EvaluationRequest,
    EvaluationResponse
)

logger = setup_logging("evaluation")

app = FastAPI(title="Evaluation Service")
register_exception_handlers(app)

evaluation_service = EvaluationService()


@app.get("/health")
def health():
    return {"status": "ok", "service": "evaluation"}


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest):
    """
    Evaluate a retrieval model.
    
    Calculates standard IR metrics:
    - MAP (Mean Average Precision)
    - nDCG (Normalized Discounted Cumulative Gain)
    - Recall@k
    - Precision@k
    """
    try:
        logger.info(f"Received evaluation request: model={request.model}")
        return evaluation_service.evaluate(request)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets")
def get_datasets():
    """Get list of available datasets for evaluation"""
    return {
        "datasets": evaluation_service.get_available_datasets()
    }


@app.get("/status")
def get_status():
    """Get service status and configuration"""
    return {
        "service": "evaluation",
        "status": "ready",
        "supported_models": [
            "tfidf",
            "embedding",
            "bm25",
            "hybrid_serial",
            "hybrid_parallel"
        ],
        "supported_metrics": [
            "MAP",
            "nDCG",
            "Recall",
            "Precision"
        ]
    }


@app.get("/history")
def get_history(model: str = None, dataset: str = None, limit: int = 10):
    """Get evaluation history"""
    try:
        return {
            "history": evaluation_service.get_evaluation_history(
                model=model,
                dataset=dataset,
                limit=limit
            )
        }
    except Exception as e:
        logger.error(f"Failed to get history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare")
def compare_models(dataset: str, top_k: int = 100):
    """Compare all models for a specific dataset"""
    try:
        return {
            "dataset": dataset,
            "top_k": top_k,
            "models": evaluation_service.compare_models(dataset=dataset, top_k=top_k)
        }
    except Exception as e:
        logger.error(f"Failed to compare models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """
    Get latest evaluation metrics for all models.
    Returns metrics in format expected by frontend.
    """
    try:
        results = evaluation_service.get_latest_metrics()
        return results
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/per-query")
def get_per_query_analysis():
    """Get best/worst performing queries per model"""
    try:
        return evaluation_service.get_per_query_analysis()
    except Exception as e:
        logger.error(f"Failed to get per-query analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/refinement-comparison")
def get_refinement_comparison():
    """Get metrics comparison before/after query refinement"""
    try:
        return evaluation_service.get_refinement_comparison()
    except Exception as e:
        logger.error(f"Failed to get refinement comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run")
def run_evaluation(
    model: str,
    dataset: str = "beir/quora/test",
    top_k: int = 100,
    query_limit: int = None
):
    """
    Run evaluation and save results.
    Simplified endpoint for frontend integration.
    """
    try:
        request = EvaluationRequest(
            dataset_name=dataset,
            model=model,
            top_k=top_k,
            query_limit=query_limit,
            include_per_query=False
        )
        return evaluation_service.evaluate(request)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation run failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
