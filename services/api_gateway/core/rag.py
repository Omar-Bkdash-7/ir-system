"""Pass-through to the RAG service.

The gateway does not interpret RAG payloads — it forwards them and reproduces the
answer (or the failure) as it came back. It owns two things: the timeouts (a
generation can take minutes on CPU while a health probe must not hang the UI) and
one reusable HTTP client, so a chat turn does not pay a new connection and TLS
handshake for every call.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from common.http_client import DownstreamError
from services.api_gateway.config import GatewaySettings

logger = logging.getLogger(__name__)


class RagProxy:
    """Async client for the RAG service."""

    def __init__(self, settings: GatewaySettings, client: Optional[httpx.AsyncClient] = None) -> None:
        self._settings = settings
        self._client = client
        self._lock = asyncio.Lock()

    async def _http(self) -> httpx.AsyncClient:
        """The shared client, created on first use (it must belong to the running loop)."""
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    # Timeouts are per request: generation and probes differ by an order of magnitude.
                    timeout=None,
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                )
        return self._client

    async def aclose(self) -> None:
        """Close the shared client (called from the app lifespan)."""
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()

    # --------------------------------------------------------------------------------------------- routes
    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/generate", timeout=self._settings.rag_timeout, body=payload)

    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/health", timeout=self._settings.rag_probe_timeout)

    async def models(self) -> Dict[str, Any]:
        return await self._request("GET", "/models", timeout=self._settings.rag_models_timeout)

    async def local_model(self) -> Dict[str, Any]:
        return await self._request("GET", "/local-model", timeout=self._settings.rag_probe_timeout)

    async def download_local_model(self) -> Dict[str, Any]:
        # The RAG service starts the download in the background and answers at once.
        return await self._request("POST", "/local-model/download", timeout=self._settings.rag_probe_timeout)

    # --------------------------------------------------------------------------------------------- internals
    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._settings.rag_url}{path}"
        try:
            client = await self._http()
            response = await client.request(method, url, json=body, timeout=timeout)
        except httpx.HTTPError as exc:
            logger.warning("RAG %s %s unreachable: %s", method, path, exc)
            raise DownstreamError("rag", f"{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            detail = (response.text or "").strip()[:300] or f"HTTP {response.status_code}"
            logger.warning("RAG %s %s failed: HTTP %d", method, path, response.status_code)
            raise DownstreamError("rag", detail, status_code=response.status_code)

        try:
            return response.json()
        except ValueError as exc:
            raise DownstreamError("rag", f"response from {path} is not JSON",
                                  status_code=502) from exc
