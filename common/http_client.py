"""The HTTP client the services use to call each other.

Every outbound call in the gateway and the evaluation service goes through here:
one JSON request, a timeout from settings, a bounded retry on transient failures,
and a single exception type that says which service failed.

Sessions are kept per thread: ``requests.Session`` is a connection pool with
mutable state and is not safe to share between threads (evaluation runs queries
in parallel, and FastAPI serves endpoints on a thread pool).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import requests

from common.exceptions import ServiceError

logger = logging.getLogger(__name__)

#: Failures worth retrying: nothing answered, or the service is momentarily unwell.
_TRANSIENT = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


class DownstreamError(ServiceError):
    """A service we depend on did not answer usably."""

    def __init__(self, service: str, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(f"{service}: {message}")
        self.service = service
        self.status_code = status_code
        #: The message without the service prefix, which is what an HTTP client should see.
        self.detail = message


class JsonClient:
    """Small JSON-over-HTTP client with one retry policy for the whole process."""

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        *,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
    ) -> None:
        self._shared_session = session
        self._local = threading.local()
        self._max_retries = max(1, int(max_retries))
        self._backoff = max(0.0, float(backoff_seconds))

    def post(self, url: str, payload: Dict[str, Any], *, timeout: float, service: str) -> Dict[str, Any]:
        return self._send("POST", url, payload, timeout=timeout, service=service)

    def get(self, url: str, *, timeout: float, service: str,
            params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._send("GET", url, None, timeout=timeout, service=service, params=params)

    def close(self) -> None:
        """Close this thread's session, and the injected one if there was one."""
        if self._shared_session is not None:
            self._shared_session.close()
        session = getattr(self._local, "session", None)
        if session is not None:
            session.close()
            del self._local.session

    # --------------------------------------------------------------------------------------------- internals
    def _session(self) -> requests.Session:
        """The injected session (tests, single-threaded) or this thread's own."""
        if self._shared_session is not None:
            return self._shared_session
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = requests.Session()
        return session

    def _send(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]],
        *,
        timeout: float,
        service: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        backoff = self._backoff
        last: Optional[BaseException] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session().request(method, url, json=payload, params=params, timeout=timeout)
            except _TRANSIENT as exc:
                last = exc
                logger.warning("%s %s unreachable (attempt %d/%d): %s",
                               service, url, attempt, self._max_retries, exc)
            else:
                if response.status_code < 400:
                    return self._decode(response, service, url)

                detail = self._error_detail(response)
                if response.status_code < 500:
                    # The service understood us and refused: retrying cannot help.
                    raise DownstreamError(service, f"HTTP {response.status_code}: {detail}",
                                          status_code=response.status_code)

                last = DownstreamError(service, f"HTTP {response.status_code}: {detail}",
                                       status_code=response.status_code)
                logger.warning("%s %s returned HTTP %d (attempt %d/%d)",
                               service, url, response.status_code, attempt, self._max_retries)

            if attempt < self._max_retries and backoff > 0:
                time.sleep(backoff)
                backoff *= 2

        if isinstance(last, DownstreamError):
            raise last
        raise DownstreamError(service, f"unreachable after {self._max_retries} attempts: {last}")

    @staticmethod
    def _decode(response: requests.Response, service: str, url: str) -> Dict[str, Any]:
        if not response.content:
            return {}
        try:
            body = response.json()
        except ValueError as exc:
            raise DownstreamError(service, f"response from {url} is not JSON",
                                  status_code=response.status_code) from exc
        return body if isinstance(body, dict) else {"data": body}

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return (response.text or "").strip()[:200] or "no detail"
        if isinstance(body, dict):
            return str(body.get("detail") or body.get("message") or body.get("error") or body)[:200]
        return str(body)[:200]
