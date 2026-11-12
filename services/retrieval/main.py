"""Retrieval service: the HTTP layer. It only validates requests and calls services.retrieval.core.service."""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Request

from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
from services.retrieval.core.service import RetrievalService
from services.retrieval.schemas import (
    HybridParallelRequest,
    HybridSerialRequest,
    RetrievalRequest,
    RetrievalResponse,
)

logger = setup_logging("retrieval")


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RetrievalService.create()
    app.state.service = service
    service.start_warmup()  # only does something when RETRIEVAL_WARMUP / retrieval.warmup is set
    logger.info("Retrieval service started (indexes load on first use unless a warm-up list is configured)")
    try:
        yield
    finally:
        service.close()
        logger.info("Retrieval service stopped")


app = FastAPI(title="Retrieval Service", lifespan=lifespan)
register_exception_handlers(app)


def get_service(request: Request) -> RetrievalService:
    service = getattr(request.app.state, "service", None)
    if service is None:  # e.g. a TestClient used without running the lifespan
        service = request.app.state.service = RetrievalService.create()
    return service


@app.get("/health")
def health():
    return {"status": "ok", "service": "retrieval"}


# ----------------------------------------------------------------------------------------------- retrieval
@app.post("/retrieve", response_model=RetrievalResponse, response_model_exclude_none=True)
def retrieve_endpoint(req: RetrievalRequest, service: RetrievalService = Depends(get_service)):
    return service.retrieve(req).to_response()


@app.post("/retrieve/tfidf", response_model=RetrievalResponse, response_model_exclude_none=True)
def retrieve_tfidf_endpoint(req: RetrievalRequest, service: RetrievalService = Depends(get_service)):
    return service.retrieve(req, model="tfidf").to_response()


@app.post("/retrieve/embedding", response_model=RetrievalResponse, response_model_exclude_none=True)
def retrieve_embedding_endpoint(req: RetrievalRequest, service: RetrievalService = Depends(get_service)):
    return service.retrieve(req, model="embedding").to_response()


@app.post("/retrieve/bm25", response_model=RetrievalResponse, response_model_exclude_none=True)
def retrieve_bm25_endpoint(req: RetrievalRequest, service: RetrievalService = Depends(get_service)):
    return service.retrieve(req, model="bm25").to_response()


@app.post("/retrieve/hybrid/serial", response_model=RetrievalResponse, response_model_exclude_none=True)
def retrieve_hybrid_serial_endpoint(req: HybridSerialRequest, service: RetrievalService = Depends(get_service)):
    return service.hybrid_serial(req).to_response()


@app.post("/retrieve/hybrid/parallel", response_model=RetrievalResponse, response_model_exclude_none=True)
def retrieve_hybrid_parallel_endpoint(req: HybridParallelRequest, service: RetrievalService = Depends(get_service)):
    return service.hybrid_parallel(req).to_response()


# ----------------------------------------------------------------------------------------------- status
@app.get("/status")
def overall_status(dataset: Optional[str] = None, service: RetrievalService = Depends(get_service)):
    return service.overall_status(dataset)


# --- shared HTTP protections -------------------------------------------------
from common.security import install_cors  # noqa: E402

install_cors(app)  # installed last, so the policy also covers throttled responses
