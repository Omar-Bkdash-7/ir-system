"""Remote health: circuit breaker + cached, non-blocking connectivity probe.

- The router asks `is_open()` to skip a remote that just failed (no 5s timeout on every request).
- `/models` calls `snapshot()` which ALWAYS returns instantly (stale-while-revalidate): when the
  cached value is old, a single background probe is scheduled.
- The probe is `GET {base_url}/models` (free, no inference credits, also verifies DNS + TLS).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Optional

import httpx

from common.logging_config import setup_logging
from services.rag.config import RAGSettings
from services.rag.errors import GeneratorError
from services.rag.generators.remote_generator import classify_http_error, classify_transport_error

logger = setup_logging("rag.remote_monitor")

# error code -> public state shown in /models
_STATE_BY_CODE = {
    "no_token": "not_configured",
    "auth_error": "auth_error",
    "quota_exhausted": "quota_exhausted",
    "model_unavailable": "model_unavailable",
}


class RemoteMonitor:
    def __init__(self, settings: RAGSettings, client: httpx.AsyncClient, clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self.client = client
        self._clock = clock
        self._open_until = 0.0
        self.state = "not_configured" if not settings.remote_configured else "unknown"
        self.detail: Optional[str] = None
        self.code: Optional[str] = None
        self.model_listed: Optional[bool] = None
        self._checked_at: Optional[float] = None
        self._probe_task: Optional["asyncio.Task[None]"] = None

    # ---- circuit breaker ---------------------------------------------------
    def is_open(self) -> bool:
        return self._clock() < self._open_until

    def retry_in(self) -> int:
        return max(0, int(self._open_until - self._clock()))

    def record_success(self) -> None:
        self._open_until = 0.0
        self.state, self.code, self.detail = "ok", None, None
        self._checked_at = self._clock()

    def record_failure(self, err: GeneratorError) -> None:
        cooldown = self.settings.cooldown_persistent if err.persistent else self.settings.cooldown_transient
        self._open_until = max(self._open_until, self._clock() + cooldown)
        self.code = err.code
        self.state = _STATE_BY_CODE.get(err.code, "unreachable")
        self.detail = err.message
        self._checked_at = self._clock()

    # ---- probe ---------------------------------------------------------------
    async def probe(self) -> None:
        if not self.settings.remote_configured:
            self.state, self.code, self.detail = "not_configured", "no_token", "No API token configured."
            self._checked_at = self._clock()
            return
        try:
            response = await self.client.get("/models", timeout=httpx.Timeout(5.0))
        except httpx.HTTPError as exc:
            err = classify_transport_error(exc)
            self.state, self.code, self.detail = "unreachable", err.code, err.message
        else:
            if response.status_code == 200:
                self.state, self.code, self.detail = "ok", None, None
                self.model_listed = self._model_listed(response)
                if self.model_listed is False:
                    self.detail = f"Model '{self.settings.remote_model_id}' is not in the provider catalog."
            elif response.status_code in (401, 403):
                err = classify_http_error(response.status_code, "")
                self.state, self.code, self.detail = "auth_error", err.code, err.message
            else:
                self.state, self.code, self.detail = "unreachable", "server_error", f"HTTP {response.status_code}"
        self._checked_at = self._clock()

    def _model_listed(self, response: httpx.Response) -> Optional[bool]:
        try:
            ids = {m.get("id") for m in response.json().get("data", []) if isinstance(m, dict)}
        except Exception:
            return None
        return self.settings.remote_model_id in ids if ids else None

    def _stale(self) -> bool:
        if self._checked_at is None:
            return True
        ttl = self.settings.probe_ttl_ok if self.state == "ok" else self.settings.probe_ttl_fail
        return (self._clock() - self._checked_at) > ttl

    async def _probe_safely(self) -> None:
        try:
            await self.probe()
        except Exception:  # a probe must never crash the service
            logger.exception("remote probe crashed")

    def snapshot(self) -> Dict[str, Any]:
        """Instant, never raises, never blocks. Schedules a background refresh when stale."""
        if self.settings.remote_configured and self._stale() and not self.is_open():
            if self._probe_task is None or self._probe_task.done():
                try:
                    self._probe_task = asyncio.get_running_loop().create_task(self._probe_safely())
                except RuntimeError:
                    pass  # no running loop (sync context) -> skip
        state = self.state
        if self.is_open() and self.code:
            state = _STATE_BY_CODE.get(self.code, "unreachable")
        return {
            "id": self.settings.remote_model,
            "provider_host": self.settings.remote_host,
            "type": "remote",
            "configured": self.settings.remote_configured,
            "state": state,
            "code": self.code,
            "detail": self.detail,
            "model_listed": self.model_listed,
            "retry_in_s": self.retry_in() if self.is_open() else 0,
            "available": state in ("ok", "unknown"),
        }
