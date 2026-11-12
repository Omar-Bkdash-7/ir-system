"""API gateway: the HTTP layer. Routing and validation only; the work lives in core/."""
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from common.exception_handlers import register_exception_handlers
from common.http_client import DownstreamError
from common.logging_config import setup_logging
from services.api_gateway.config import get_settings
from services.api_gateway.core.rag import RagProxy
from services.api_gateway.core.search import SearchService
from services.api_gateway.schemas import (
    BasicSearchRequest,
    HybridParallelSearchRequest,
    HybridSerialSearchRequest,
    QueryRefineRequest,
)

logger = setup_logging("api_gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.search = SearchService.create()
    app.state.rag = RagProxy(get_settings())
    logger.info("API gateway started")
    try:
        yield
    finally:
        app.state.search.close()
        await app.state.rag.aclose()
        logger.info("API gateway stopped")


app = FastAPI(title="API Gateway", lifespan=lifespan)
register_exception_handlers(app)


async def downstream_error_handler(request: Request, exc: DownstreamError) -> JSONResponse:
    """
    Reproduce the status a downstream service answered with (404 missing index, 422 bad request, ...).

    Without this, every downstream failure reached the generic handler and looked
    like a 500 from the gateway. A service that is simply unreachable has no
    status, and becomes a 502.
    """
    logger.warning("Downstream failure on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=exc.status_code or status.HTTP_502_BAD_GATEWAY,
        content={"error": type(exc).__name__, "detail": exc.detail},
    )


app.add_exception_handler(DownstreamError, downstream_error_handler)


def get_search(request: Request) -> SearchService:
    service = getattr(request.app.state, "search", None)
    if service is None:  # e.g. a TestClient used without running the lifespan
        service = request.app.state.search = SearchService.create()
    return service


def get_rag(request: Request) -> RagProxy:
    proxy = getattr(request.app.state, "rag", None)
    if proxy is None:
        proxy = request.app.state.rag = RagProxy(get_settings())
    return proxy


# ----------------------------------------------------------------------------------------------- health & data
@app.get("/health")
def health():
    return {"status": "ok", "service": "api_gateway"}


@app.get("/datasets/info")
def datasets_info(service: SearchService = Depends(get_search)):
    return service.dataset_info()


# ----------------------------------------------------------------------------------------------- query
@app.post("/query/refine")
def query_refine_endpoint(req: QueryRefineRequest, service: SearchService = Depends(get_search)):
    if not req.query or not req.query.strip():
        raise ValueError("Query cannot be empty")
    return service.refine(
        req.query,
        req.history,
        req.enabled,
        req.session_id,
        req.is_evaluation,
        req.dataset,
    )


# ----------------------------------------------------------------------------------------------- search
@app.post("/search")
def search_endpoint(req: BasicSearchRequest, service: SearchService = Depends(get_search)):
    return service.search_basic(req)


@app.post("/search/hybrid/serial")
def search_hybrid_serial_endpoint(req: HybridSerialSearchRequest, service: SearchService = Depends(get_search)):
    return service.search_hybrid_serial(req)


@app.post("/search/hybrid/parallel")
def search_hybrid_parallel_endpoint(req: HybridParallelSearchRequest, service: SearchService = Depends(get_search)):
    return service.search_hybrid_parallel(req)


# ----------------------------------------------------------------------------------------------- rag
@app.post("/rag/generate")
async def rag_generate_endpoint(request: Dict[str, Any], rag: RagProxy = Depends(get_rag)):
    return await rag.generate(request)


@app.get("/rag/health")
async def rag_health_endpoint(rag: RagProxy = Depends(get_rag)):
    return await rag.health()


@app.get("/rag/models")
async def rag_models_endpoint(rag: RagProxy = Depends(get_rag)):
    return await rag.models()


@app.get("/rag/local-model")
async def rag_local_model_status_endpoint(rag: RagProxy = Depends(get_rag)):
    return await rag.local_model()


@app.post("/rag/local-model/download")
async def rag_local_model_download_endpoint(rag: RagProxy = Depends(get_rag)):
    return await rag.download_local_model()


# --- shared HTTP protections -------------------------------------------------
from common.security import install_cors, install_rate_limit  # noqa: E402

install_rate_limit(app)  # only the public entry point is throttled
install_cors(app)  # installed last, so the policy also covers throttled responses
