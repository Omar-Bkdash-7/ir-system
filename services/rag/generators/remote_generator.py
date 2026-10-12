"""Remote generator for ANY OpenAI-compatible /chat/completions endpoint.

Default target: Hugging Face Inference Providers router (https://router.huggingface.co/v1).
The legacy api-inference.huggingface.co host is retired and must never be used.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import httpx

from common.logging_config import setup_logging
from services.rag.config import RAGSettings
from services.rag.errors import GeneratorError
from services.rag.netutil import has_dns_cause
from services.rag.generators.base_generator import BaseGenerator, GenerationRequest, GenerationResult
from services.rag.prompt_builder import build_messages, strip_reasoning

logger = setup_logging("rag.remote_generator")

_RETRYABLE_STATUS = {500, 502, 503, 504}


def build_remote_client(settings: RAGSettings, transport: Optional[httpx.AsyncBaseTransport] = None) -> httpx.AsyncClient:
    """One shared client for the whole process (connection pooling). Created in the FastAPI lifespan."""
    headers = {"Content-Type": "application/json"}
    if settings.remote_api_key:
        headers["Authorization"] = f"Bearer {settings.remote_api_key}"
    return httpx.AsyncClient(
        base_url=settings.remote_base_url,
        headers=headers,
        timeout=httpx.Timeout(
            connect=settings.remote_connect_timeout,
            read=settings.remote_read_timeout,
            write=10.0,
            pool=5.0,
        ),
        transport=transport,
    )


def classify_transport_error(exc: BaseException) -> GeneratorError:
    if isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        return GeneratorError("remote", "timeout", "The remote model did not answer in time.")
    if isinstance(exc, httpx.ConnectError):
        if has_dns_cause(exc):
            return GeneratorError(
                "remote", "dns_error", "DNS resolution failed for the remote host.",
                hint="Check RAG_REMOTE_BASE_URL (legacy api-inference.huggingface.co no longer exists) and run GET /diagnostics/network.",
            )
        return GeneratorError("remote", "connect_error", "Could not connect to the remote host.")
    return GeneratorError("remote", "connect_error", f"Network error talking to the remote host: {type(exc).__name__}")


def _error_text(response: httpx.Response) -> str:
    try:
        body: Any = response.json()
        err = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(err, dict):
            err = err.get("message") or err
        return str(err)[:300]
    except Exception:
        return (response.text or "")[:300]


def classify_http_error(status: int, detail: str) -> GeneratorError:
    low = detail.lower()
    if status in (401, 403):
        return GeneratorError(
            "remote", "auth_error", f"The provider rejected the API token (HTTP {status}).",
            hint="Create a fine-grained Hugging Face token with 'Make calls to Inference Providers' and set HUGGINGFACE_API_TOKEN.",
        )
    if status == 402:
        return GeneratorError(
            "remote", "quota_exhausted", "Free monthly credits are exhausted (HTTP 402).",
            hint="Wait for the monthly reset, add credits, or use the local model.",
        )
    if status == 429:
        return GeneratorError("remote", "rate_limited", "Rate limit reached (HTTP 429).")
    if status in (400, 404, 422):
        if "model" in low:
            return GeneratorError(
                "remote", "model_unavailable", f"The remote model is not available (HTTP {status}).",
                hint="Pick another model from GET https://router.huggingface.co/v1/models and set RAG_REMOTE_MODEL.",
            )
        return GeneratorError("remote", "bad_request", f"The provider rejected the request (HTTP {status}): {detail}")
    if status >= 500:
        return GeneratorError("remote", "server_error", f"Provider error (HTTP {status}).")
    return GeneratorError("remote", "bad_response", f"Unexpected HTTP {status}: {detail}")


def _extract_text(data: Any) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise GeneratorError("remote", "bad_response", "Malformed completion payload from the provider.")
    if isinstance(content, list):  # OpenAI "parts" format
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    text = strip_reasoning(content or "")
    if not text:
        raise GeneratorError("remote", "bad_response", "The model returned an empty answer.")
    return text


class RemoteGenerator(BaseGenerator):
    source = "remote"

    def __init__(self, settings: RAGSettings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    def unavailable_reason(self) -> Optional[str]:
        if not self.settings.remote_configured:
            return "No API token configured (set HUGGINGFACE_API_TOKEN)."
        return None

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        reason = self.unavailable_reason()
        if reason:
            raise GeneratorError("remote", "no_token", reason)

        cfg = request.config
        messages = build_messages(
            request.query, request.documents,
            max_context_chars=self.settings.remote_max_context_chars,
            system_prompt=cfg.system_prompt, use_context=cfg.use_context,
        )
        payload = {
            "model": self.settings.remote_model,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": 0.95,
            "stream": False,
        }

        started = time.perf_counter()
        attempt = 0
        while True:
            try:
                response = await self.client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:  # connect / DNS / timeout / protocol
                logger.warning("remote call failed host=%s error=%s", self.settings.remote_host, type(exc).__name__)
                raise classify_transport_error(exc) from exc

            if response.status_code in _RETRYABLE_STATUS and attempt < self.settings.remote_max_retries:
                attempt += 1
                logger.warning("remote HTTP %s, retry %s", response.status_code, attempt)
                await asyncio.sleep(1.0 * attempt)
                continue
            break

        if response.status_code != 200:
            detail = _error_text(response)
            logger.warning("remote HTTP %s model=%s detail=%s", response.status_code, self.settings.remote_model, detail)
            raise classify_http_error(response.status_code, detail)

        try:
            data = response.json()
        except ValueError:
            raise GeneratorError("remote", "bad_response", "The provider returned invalid JSON.")

        text = _extract_text(data)
        return GenerationResult(
            text=text,
            source="remote",
            model=str(data.get("model") or self.settings.remote_model_id),
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=data.get("usage"),
        )
