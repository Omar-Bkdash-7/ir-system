"""
Base HTTP client for every IR System microservice call.

This is the ONLY module in the codebase allowed to import `requests`.
Every other module (pages, components, the rest of `api/`) goes through
`get()` / `post()` / `health_check()` here, which normalizes every possible
outcome -- success, 4xx, 5xx, connection refused, timeout -- into a single
`APIResponse` object so callers never need a try/except around a raw
`requests` call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

import config


@dataclass
class APIResponse:
    """
    Normalized result of an API call, regardless of outcome.

    `reason` is the coarse failure class so callers can branch without parsing
    the message: "client" (4xx), "server" (5xx), "connection", "timeout" or
    "request". It is None on success.
    """

    ok: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    elapsed_ms: float = 0.0
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


def _extract_error_detail(response: requests.Response) -> str:
    """Best-effort extraction of a human-readable error from a failed response."""
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("detail") or body.get("message") or body.get("error") or body)[:200]
        return str(body)[:200]
    except ValueError:
        text = (response.text or "").strip()
        return text[:200] if text else f"HTTP {response.status_code}"


def _request(method: str, url: str, *, timeout: Optional[int] = None,
             max_retries: Optional[int] = None, **kwargs: Any) -> APIResponse:
    """Shared retry/backoff/error-normalization core for GET and POST."""
    timeout = timeout if timeout is not None else config.REQUEST_TIMEOUT
    max_retries = max_retries if max_retries is not None else config.MAX_RETRIES
    max_retries = max(1, max_retries)
    backoff = config.RETRY_BACKOFF_BASE_SECONDS
    last_error = "Unknown error"
    reason = "request"
    started = time.perf_counter()

    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000

            if response.status_code >= 500:
                last_error = f"Service error (HTTP {response.status_code})"
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return APIResponse(False, None, last_error, response.status_code,
                                   elapsed_ms, reason="server")

            if response.status_code >= 400:
                return APIResponse(False, None, _extract_error_detail(response),
                                    response.status_code, elapsed_ms, reason="client")

            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            return APIResponse(True, payload, None, response.status_code, elapsed_ms)

        except requests.exceptions.ConnectionError:
            last_error = "Connection refused — the service may be offline"
            reason = "connection"
        except requests.exceptions.Timeout:
            last_error = f"Request timed out after {timeout}s"
            reason = "timeout"
        except requests.exceptions.RequestException as exc:  # noqa: BLE001
            last_error = f"Request error: {exc}"
            reason = "request"

        if attempt < max_retries - 1:
            time.sleep(backoff)
            backoff *= 2

    elapsed_ms = (time.perf_counter() - started) * 1000
    return APIResponse(False, None, last_error, None, elapsed_ms, reason=reason)


def get(url: str, *, params: Optional[dict] = None, timeout: Optional[int] = None,
        max_retries: Optional[int] = None) -> APIResponse:
    """Perform a GET request through the shared retry/error pipeline."""
    return _request("GET", url, params=params, timeout=timeout, max_retries=max_retries)


def post(url: str, *, json: Optional[dict] = None, params: Optional[dict] = None,
          timeout: Optional[int] = None, max_retries: Optional[int] = None) -> APIResponse:
    """Perform a POST request through the shared retry/error pipeline."""
    return _request("POST", url, json=json, params=params, timeout=timeout, max_retries=max_retries)


def delete(url: str, *, params: Optional[dict] = None, timeout: Optional[int] = None,
           max_retries: Optional[int] = None) -> APIResponse:
    """Perform a DELETE request through the shared retry/error pipeline."""
    return _request("DELETE", url, params=params, timeout=timeout, max_retries=max_retries)


def health_check(base_url: str) -> APIResponse:
    """Single-attempt, short-timeout probe used for liveness badges."""
    return get(f"{base_url}/health", timeout=config.HEALTH_TIMEOUT, max_retries=1)