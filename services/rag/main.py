from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from common.exception_handlers import register_exception_handlers
from common.exceptions import ServiceError
from common.logging_config import setup_logging
from services.rag.config import RAGSettings
from services.rag.diagnostics import run_download_probe, run_network_diagnostics
from services.rag.errors import GeneratorError
from services.rag.generators.base_generator import GenerationRequest
from services.rag.generators.local_generator import LocalGenerator
from services.rag.generators.remote_generator import RemoteGenerator, build_remote_client
from services.rag.generators.router import GeneratorRouter
from services.rag.local_model.manager import LocalModelManager
from services.rag.rag_service import RAGService
from services.rag.remote_monitor import RemoteMonitor
from services.rag.schemas import GenerationConfig, RAGRequest, RAGResponse

logger = setup_logging("rag")
VERSION = "2.0.0"

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def _size_hint(model_id: str) -> str:
    match = _SIZE_RE.search(model_id.split("/")[-1])
    return f"{match.group(1)}B parameters" if match else "hosted model"


def create_app(
    settings: Optional[RAGSettings] = None,
    remote_transport: Optional[httpx.AsyncBaseTransport] = None,
    retrieval_transport: Optional[httpx.AsyncBaseTransport] = None,
) -> FastAPI:
    """Factory so tests can inject settings / mocked transports. `app = create_app()` is what uvicorn loads."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = settings or RAGSettings.from_env()
        remote_client = build_remote_client(cfg, transport=remote_transport)
        retrieval_client = httpx.AsyncClient(timeout=cfg.retrieval_timeout, transport=retrieval_transport)
        manager = LocalModelManager(cfg)
        monitor = RemoteMonitor(cfg, remote_client)
        router = GeneratorRouter(cfg, RemoteGenerator(cfg, remote_client), LocalGenerator(cfg, manager), monitor)
        app.state.settings, app.state.manager, app.state.monitor, app.state.router = cfg, manager, monitor, router
        app.state.rag = RAGService(cfg, router, retrieval_client)
        manager.resume_if_interrupted()  # continue a download the user started before a restart/reload
        logger.info(
            "RAG %s ready | remote host=%s model=%s token_configured=%s | local model=%s state=%s dir=%s | default_mode=%s",
            VERSION, cfg.remote_host, cfg.remote_model, cfg.remote_configured,
            cfg.local_model, manager.state, cfg.models_dir, cfg.default_mode,
        )
        try:
            yield
        finally:
            await manager.shutdown()
            await remote_client.aclose()
            await retrieval_client.aclose()

    app = FastAPI(
        title="RAG Service",
        description="Retrieval-Augmented Generation: remote LLM (OpenAI-compatible, default Hugging Face router) with local CPU fallback",
        version=VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    register_exception_handlers(app)

    @app.get("/health")
    async def health(request: Request):
        # Liveness only: no network, no model loading, no probe scheduling. Docker's healthcheck calls this.
        # NOTE: keep these handlers `async def`: RemoteMonitor.snapshot() schedules a background probe and
        # needs a running event loop (plain `def` handlers run in a threadpool thread WITHOUT one).
        st = request.app.state
        return {
            "status": "healthy",
            "service": "rag",
            "version": VERSION,
            "dependencies": {  # legacy keys kept for the UI / gateway; all values are LOCAL state
                "retrieval": st.settings.retrieval_url,
                "huggingface_token": "configured" if st.settings.remote_configured else "not_configured",
                "remote_model": st.settings.remote_model_id,
                "remote_available": st.settings.remote_configured and st.monitor.state in ("ok", "unknown"),
                "local_model": st.manager.state,
            },
        }

    @app.get("/models")
    async def list_models(request: Request):
        st = request.app.state
        remote = st.monitor.snapshot()
        local = st.manager.status()
        remote_ok = remote["configured"]
        modes = [
            {"id": "auto", "label": "Auto (remote, fallback to local)", "available": remote_ok or local["available"],
             "reason": None if (remote_ok or local["available"]) else "No token and no local model"},
            {"id": "remote", "label": "Remote only", "available": remote_ok,
             "reason": None if remote_ok else "HUGGINGFACE_API_TOKEN is not configured"},
            {"id": "local", "label": "Local only", "available": local["available"],
             "reason": None if local["available"] else "Local model is not downloaded"},
        ]
        remote.update({  # legacy keys: id / name / size / type / quality / speed / requirements / available
            "name": st.settings.remote_model_id.split("/")[-1],
            "size": _size_hint(st.settings.remote_model_id),
            "quality": "High",
            "speed": "Fast (network dependent)",
            "requirements": "Internet + Hugging Face token with 'Make calls to Inference Providers'",
        })
        return {
            "default_mode": st.settings.default_mode,
            "modes": modes,
            "remote_model": remote,
            "local_model": local,
            "retrieval_models": ["tfidf", "embedding", "bm25", "hybrid"],
        }

    @app.get("/local-model")
    async def local_model_status(request: Request):
        return request.app.state.manager.status()

    @app.post("/local-model/download", status_code=202)
    async def local_model_download(request: Request, response: Response):
        status = await request.app.state.manager.start_download()
        if status["state"] != "downloading":
            response.status_code = 200  # already ready / disabled / failed instantly
        return status

    @app.post("/generate", response_model=RAGResponse)
    async def generate_answer(payload: RAGRequest, request: Request):
        if not payload.query or not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        if not payload.dataset or not payload.dataset.strip():
            raise HTTPException(status_code=400, detail="Dataset must be specified")
        logger.info("RAG request: query=%r dataset=%s mode=%s", payload.query[:50], payload.dataset,
                    payload.generation_config.mode or request.app.state.settings.default_mode)
        try:
            return await request.app.state.rag.generate_answer(payload)
        except (HTTPException, ServiceError):
            raise  # ServiceError (e.g. retrieval down) is mapped by register_exception_handlers
        except Exception as exc:  # anything else is a bug: log it, do not leak internals
            logger.error("Error processing RAG request: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error while processing the RAG request")

    @app.get("/diagnostics/download")
    async def diagnostics_download(request: Request):
        st = request.app.state
        if not st.settings.diagnostics_enabled:
            raise HTTPException(status_code=404, detail="Diagnostics disabled (set RAG_ENABLE_DIAGNOSTICS=true)")
        return await run_download_probe(st.settings, st.manager)

    @app.get("/diagnostics/network")
    async def diagnostics_network(request: Request, probe_inference: bool = Query(False)):
        st = request.app.state
        if not st.settings.diagnostics_enabled:
            raise HTTPException(status_code=404, detail="Diagnostics disabled (set RAG_ENABLE_DIAGNOSTICS=true)")
        report = await run_network_diagnostics(st.settings, st.manager)
        if probe_inference:  # costs ~0 credits (3 tokens) but proves token + permission + credits end to end
            try:
                result = await st.router.generators["remote"].generate(
                    GenerationRequest("Reply with: OK", [], GenerationConfig(max_tokens=3, temperature=0.0, use_context=False))
                )
                report["inference_probe"] = {"ok": True, "model": result.model, "latency_ms": result.latency_ms}
            except GeneratorError as exc:
                report["inference_probe"] = {"ok": False, "error": exc.to_dict()}
            except Exception as exc:
                report["inference_probe"] = {"ok": False, "error": {"message": str(exc)}}
        return report

    return app


app = create_app()
