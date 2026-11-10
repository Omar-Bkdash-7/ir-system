"""HTTP surface of the query refinement service.

Errors are translated once, by ``common.exception_handlers`` (``ValueError`` becomes a 400). The previous handlers
caught everything and re-raised a 500 carrying ``str(exc)``, which turned a corrupt corpus into a permanent outage
*and* leaked the internal message to the client.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
from services.query_refinement.schemas import RefineRequest, RefineResponse
from services.query_refinement.services import get_service, refine_query

logger = setup_logging("query_refinement")


@asynccontextmanager
async def lifespan(_: FastAPI):
    status = get_service().feature_status()
    logger.info("Query refinement service ready: %s", status)
    yield


app = FastAPI(
    title="Query Refinement Service",
    description="Query refinement with spelling correction, WordNet synonym expansion and history-aware boosting",
    lifespan=lifespan,
)
register_exception_handlers(app)


@app.get("/health")
def health():
    """Always 200 while the process is alive; ``status`` says whether every stage is usable."""
    status = get_service().feature_status()
    features = status["features"]
    healthy = bool(features["spelling_correction"] and features["synonym_expansion"])
    return {"status": "ok" if healthy else "degraded", "service": "query_refinement", **status}


@app.post("/refine", response_model=RefineResponse)
def refine_endpoint(request: RefineRequest) -> RefineResponse:
    result = refine_query(
        request.query,
        request.history,
        request.enabled,
        request.session_id,
        request.is_evaluation,
        request.dataset,
    )
    return RefineResponse(**result)
