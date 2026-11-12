"""HTTP protections shared by every service: one CORS policy, one rate limiter.

`CORS_ORIGINS` was only honoured by a single service, and `RATE_LIMIT` existed in
`.env.example` without any code reading it. Both live here now, so every FastAPI app
gets the same behaviour from one line.

    CORS_ORIGINS        comma separated origins, default http://localhost:8501,http://localhost:3000
                        (empty value = no CORS headers)
    RATE_LIMIT          requests per window per client, default 300; 0 disables the limiter
    RATE_LIMIT_WINDOW   window in seconds, default 60

The limiter is meant for the public entry point (the API gateway). Applying it to
services that also receive internal traffic would throttle the gateway itself, since
every service-to-service call comes from one container address.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Iterable, List, Optional, Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

DEFAULT_ORIGINS: Sequence[str] = ("http://localhost:8501", "http://localhost:3000")

#: Probes and schema endpoints are never throttled: an orchestrator polling them must not be refused.
EXEMPT_PATHS: Sequence[str] = (
    "/health",
    "/healthz",
    "/readiness",
    "/liveness",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r; using %d", name, os.getenv(name), default)
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r; using %.1f", name, os.getenv(name), default)
        return default


def cors_origins(raw: Optional[str] = None) -> List[str]:
    """Split CORS_ORIGINS; an empty value means "no cross-origin access" rather than "allow everything"."""
    value = os.getenv("CORS_ORIGINS") if raw is None else raw
    if value is None:
        return list(DEFAULT_ORIGINS)
    return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]


def install_cors(app: FastAPI, origins: Optional[Iterable[str]] = None) -> None:
    """Apply one CORS policy. Call this LAST so the policy also covers 429 responses."""
    allowed = list(origins) if origins is not None else cors_origins()
    if not allowed:
        logger.info("CORS disabled (CORS_ORIGINS is empty)")
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Retry-After"],
        max_age=600,
    )
    logger.info("CORS enabled for %s", ", ".join(allowed))


class RateLimiter:
    """Fixed-window counter per client address (one window, one bucket)."""

    def __init__(self, limit: int, window: float) -> None:
        self.limit = limit
        self.window = window
        self._lock = threading.Lock()
        self._counters: dict[str, tuple[float, int]] = {}

    def hit(self, client: str) -> Optional[int]:
        """Count one request. Returns the Retry-After seconds when the client is over the limit."""
        now = time.monotonic()
        with self._lock:
            start, count = self._counters.get(client, (now, 0))
            if now - start >= self.window:
                start, count = now, 0
            count += 1
            self._counters[client] = (start, count)
            if len(self._counters) > 10_000:  # bounded memory: drop windows that already expired
                self._counters = {
                    key: value for key, value in self._counters.items() if now - value[0] < self.window
                }
            if count > self.limit:
                return max(1, int(self.window - (now - start)))
        return None


class RateLimitMiddleware:
    """Pure ASGI middleware: no request body is read, so streaming routes keep working."""

    def __init__(self, app, *, limit: int, window: float, exempt: Sequence[str]) -> None:
        self.app = app
        self.limiter = RateLimiter(limit, window)
        self.exempt = tuple(exempt)
        self.window = window

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or self._is_exempt(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        client = (scope.get("client") or ("unknown", 0))[0]
        retry_after = self.limiter.hit(client)
        if retry_after is not None:
            body = json.dumps(
                {
                    "error": "RateLimitExceeded",
                    "detail": f"Too many requests: at most {self.limiter.limit} per {int(self.window)}s",
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"retry-after", str(retry_after).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)

    def _is_exempt(self, path: str) -> bool:
        return any(path == exempt or path.startswith(exempt + "/") for exempt in self.exempt)


def install_rate_limit(
    app: FastAPI,
    limit: Optional[int] = None,
    window: Optional[float] = None,
    exempt: Optional[Sequence[str]] = None,
) -> None:
    """Throttle by client address. `limit=0` (RATE_LIMIT=0) disables it."""
    resolved_limit = _int_env("RATE_LIMIT", 300) if limit is None else limit
    if resolved_limit <= 0:
        logger.info("Rate limiting disabled")
        return
    resolved_window = _float_env("RATE_LIMIT_WINDOW", 60.0) if window is None else window
    app.add_middleware(
        RateLimitMiddleware,
        limit=resolved_limit,
        window=max(0.001, resolved_window),
        exempt=tuple(EXEMPT_PATHS if exempt is None else exempt),
    )
    logger.info("Rate limiting enabled: %d requests per %ds per client", resolved_limit, int(resolved_window))
