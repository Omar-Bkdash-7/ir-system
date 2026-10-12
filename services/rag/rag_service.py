"""Orchestrator: retrieve context -> route generation -> build a response that never lies.

Contract: if retrieval worked but generation did not, we still return the documents with
metadata.status == "generation_failed" (HTTP 200) so the IR part of the project stays useful.
"""
from __future__ import annotations

import time
from typing import List

import httpx

from common.exceptions import ServiceError
from common.logging_config import setup_logging
from services.rag.config import RAGSettings
from services.rag.errors import GenerationUnavailableError
from services.rag.generators.base_generator import GenerationRequest
from services.rag.generators.router import GeneratorRouter
from services.rag.schemas import ContextDocument, RAGRequest, RAGResponse

logger = setup_logging("rag.service")


class RAGService:
    def __init__(self, settings: RAGSettings, router: GeneratorRouter, retrieval_client: httpx.AsyncClient):
        self.settings = settings
        self.router = router
        self.retrieval_client = retrieval_client

    async def retrieve_context(self, request: RAGRequest) -> List[ContextDocument]:
        url = f"{self.settings.retrieval_url}/retrieve"
        payload = {
            "query_text": request.query,
            "query_tokens": [],
            "dataset": request.dataset,
            "model": request.retrieval_model,
            "top_k": request.top_k,
            "include_text": True,
        }
        if request.retrieval_model == "bm25":
            payload["bm25_k1"] = request.bm25_k1
            payload["bm25_b"] = request.bm25_b
        try:
            response = await self.retrieval_client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("retrieval HTTP %s: %s", exc.response.status_code, exc.response.text[:200])
            raise ServiceError(f"Retrieval service error: {exc.response.status_code}")
        except Exception as exc:
            logger.error("retrieval failed: %s", exc, exc_info=True)
            raise ServiceError(f"Failed to retrieve context: {exc}")
        return [
            ContextDocument(
                doc_id=str(r.get("doc_id", "")),
                text=r.get("text", "") or "",
                score=float(r.get("score", 0.0) or 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in data.get("results", [])[: request.top_k]
        ]

    async def generate_answer(self, request: RAGRequest) -> RAGResponse:
        started = time.perf_counter()
        cfg = request.generation_config
        docs: List[ContextDocument] = await self.retrieve_context(request) if cfg.use_context else []

        if cfg.use_context and not docs:
            return RAGResponse(
                query=request.query, answer="No relevant documents found to answer this question.",
                context_documents=[], model_used="none", retrieval_model=request.retrieval_model,
                metadata={"status": "no_context"},
            )

        base_meta = {
            "num_docs_retrieved": len(docs),
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "requested_mode": cfg.mode or self.settings.default_mode,
        }
        try:
            result = await self.router.generate(GenerationRequest(request.query, docs, cfg), cfg.mode)
        except GenerationUnavailableError as exc:
            logger.error("generation unavailable: %s", exc)
            return RAGResponse(
                query=request.query, answer="", context_documents=docs, model_used="none",
                retrieval_model=request.retrieval_model,
                metadata={**base_meta, "status": "generation_failed", "errors": exc.to_list(),
                          "latency_ms": int((time.perf_counter() - started) * 1000)},
            )
        return RAGResponse(
            query=request.query, answer=result.text, context_documents=docs,
            model_used=f"{result.model} ({result.source})", retrieval_model=request.retrieval_model,
            metadata={**base_meta, "status": "success", "generation_source": result.source,
                      "generation_latency_ms": result.latency_ms, "usage": result.usage,
                      "fallback_used": result.fallback_used, "fallback_reason": result.fallback_reason,
                      "attempts": result.attempts,
                      "latency_ms": int((time.perf_counter() - started) * 1000)},
        )
