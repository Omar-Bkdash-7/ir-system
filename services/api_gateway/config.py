"""Runtime settings of the API gateway.

Every value has a default in code. The service URLs can also come from the shared
``services:`` section of ``configs/config.yaml`` (as they did before this module
existed), and every knob can be overridden by an environment variable, which wins.
No configuration change is required to run.

    environment variable             config key                     default
    -------------------------------  -----------------------------  ---------------------------
    GATEWAY_PREPROCESSING_URL        services.preprocessing_url     http://preprocessing:8001
    GATEWAY_RETRIEVAL_URL            services.retrieval_url         http://retrieval:8003
    GATEWAY_QUERY_REFINEMENT_URL     services.query_refinement_url  http://query_refinement:8005
    GATEWAY_RAG_URL                  services.rag_url               http://rag:8008
    GATEWAY_METADATA_DIR             paths.metadata_dir             data/metadata
    GATEWAY_PREPROCESSING_TIMEOUT    gateway.preprocessing_timeout  30
    GATEWAY_RETRIEVAL_TIMEOUT        gateway.retrieval_timeout      30
    GATEWAY_HYBRID_TIMEOUT           gateway.hybrid_timeout         60
    GATEWAY_REFINEMENT_TIMEOUT       gateway.refinement_timeout     10
    GATEWAY_RAG_TIMEOUT              gateway.rag_timeout            300
    GATEWAY_RAG_PROBE_TIMEOUT        gateway.rag_probe_timeout      10
    GATEWAY_RAG_MODELS_TIMEOUT       gateway.rag_models_timeout     30
    GATEWAY_MAX_RETRIES              gateway.max_retries            2
    GATEWAY_RETRY_BACKOFF            gateway.retry_backoff          0.2
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, TypeVar

from common.config import get_config

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _as_url(value: Any) -> str:
    url = str(value).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError("must start with http:// or https://")
    return url


def _as_positive_float(value: Any) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError("must be > 0")
    return number


def _as_positive_int(value: Any) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError("must be > 0")
    return number


def _setting(env_name: str, config_key: str, default: T, cast: Callable[[Any], T]) -> T:
    """environment variable > configs/config.yaml > default. A bad value is logged and ignored."""
    raw: Any = os.getenv(env_name)
    source = f"environment variable {env_name}"
    if raw is None or str(raw).strip() == "":
        raw = get_config(config_key, None)
        source = f"config key {config_key}"
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("Ignoring invalid value %r from %s (%s); using %r", raw, source, exc, default)
        return default


@dataclass(frozen=True)
class GatewaySettings:
    """Where the gateway forwards to, and how long it is willing to wait."""

    preprocessing_url: str = "http://preprocessing:8001"
    retrieval_url: str = "http://retrieval:8003"
    query_refinement_url: str = "http://query_refinement:8005"
    rag_url: str = "http://rag:8008"
    metadata_dir: str = "data/metadata"

    preprocessing_timeout: float = 30.0
    retrieval_timeout: float = 30.0
    hybrid_timeout: float = 60.0
    refinement_timeout: float = 10.0
    rag_timeout: float = 300.0
    rag_probe_timeout: float = 10.0
    rag_models_timeout: float = 30.0

    max_retries: int = 2
    retry_backoff: float = 0.2

    @classmethod
    def load(cls) -> "GatewaySettings":
        return cls(
            preprocessing_url=_setting("GATEWAY_PREPROCESSING_URL", "services.preprocessing_url",
                                       cls.preprocessing_url, _as_url),
            retrieval_url=_setting("GATEWAY_RETRIEVAL_URL", "services.retrieval_url",
                                   cls.retrieval_url, _as_url),
            query_refinement_url=_setting("GATEWAY_QUERY_REFINEMENT_URL", "services.query_refinement_url",
                                          cls.query_refinement_url, _as_url),
            rag_url=_setting("GATEWAY_RAG_URL", "services.rag_url", cls.rag_url, _as_url),
            metadata_dir=_setting("GATEWAY_METADATA_DIR", "paths.metadata_dir",
                                  cls.metadata_dir, str),
            preprocessing_timeout=_setting("GATEWAY_PREPROCESSING_TIMEOUT", "gateway.preprocessing_timeout",
                                           cls.preprocessing_timeout, _as_positive_float),
            retrieval_timeout=_setting("GATEWAY_RETRIEVAL_TIMEOUT", "gateway.retrieval_timeout",
                                       cls.retrieval_timeout, _as_positive_float),
            hybrid_timeout=_setting("GATEWAY_HYBRID_TIMEOUT", "gateway.hybrid_timeout",
                                    cls.hybrid_timeout, _as_positive_float),
            refinement_timeout=_setting("GATEWAY_REFINEMENT_TIMEOUT", "gateway.refinement_timeout",
                                        cls.refinement_timeout, _as_positive_float),
            rag_timeout=_setting("GATEWAY_RAG_TIMEOUT", "gateway.rag_timeout",
                                 cls.rag_timeout, _as_positive_float),
            rag_probe_timeout=_setting("GATEWAY_RAG_PROBE_TIMEOUT", "gateway.rag_probe_timeout",
                                       cls.rag_probe_timeout, _as_positive_float),
            rag_models_timeout=_setting("GATEWAY_RAG_MODELS_TIMEOUT", "gateway.rag_models_timeout",
                                        cls.rag_models_timeout, _as_positive_float),
            max_retries=max(1, _setting("GATEWAY_MAX_RETRIES", "gateway.max_retries",
                                        cls.max_retries, _as_positive_int)),
            retry_backoff=_setting("GATEWAY_RETRY_BACKOFF", "gateway.retry_backoff",
                                   cls.retry_backoff, lambda value: max(0.0, float(value))),
        )


@lru_cache(maxsize=1)
def get_settings() -> GatewaySettings:
    return GatewaySettings.load()


def reset_settings_cache() -> None:
    """For tests: forget the cached settings so the next get_settings() re-reads the environment."""
    get_settings.cache_clear()
