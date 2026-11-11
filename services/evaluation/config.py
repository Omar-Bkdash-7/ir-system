"""Runtime settings of the evaluation service.

Every value has a default in code. It can be overridden in ``configs/config.yaml``
(section ``evaluation:``) and, for the deployment-level knobs, by an environment
variable (the variable wins). No configuration change is required to run.

    key (evaluation.<key>)        environment variable                    default
    ----------------------------  --------------------------------------  -------------------------
    retrieval_url                 EVALUATION_RETRIEVAL_URL                http://retrieval:8003
    preprocessing_url             EVALUATION_PREPROCESSING_URL            http://preprocessing:8001
    query_refinement_url          EVALUATION_QUERY_REFINEMENT_URL         http://query_refinement:8005
    retrieval_timeout_seconds     EVALUATION_RETRIEVAL_TIMEOUT            60
    preprocessing_timeout_seconds EVALUATION_PREPROCESSING_TIMEOUT        10
    refinement_timeout_seconds    EVALUATION_REFINEMENT_TIMEOUT           10
    max_retries                   EVALUATION_MAX_RETRIES                  2
    retry_backoff_seconds         EVALUATION_RETRY_BACKOFF                0.5
    workers                       EVALUATION_WORKERS                      8
    default_top_k                 EVALUATION_DEFAULT_TOP_K                100
    max_top_k                     EVALUATION_MAX_TOP_K                    1000
    max_query_limit               EVALUATION_MAX_QUERY_LIMIT              1000
    save_results                  EVALUATION_SAVE_RESULTS                 true
    history_limit                 EVALUATION_HISTORY_LIMIT                100
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

#: Upper bound for `workers`: beyond this the retrieval service, not the loop, is the bottleneck.
MAX_WORKERS = 64


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a boolean: {value!r}")


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


def _as_url(value: Any) -> str:
    url = str(value).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError("must start with http:// or https://")
    return url


def _setting(env_name: str, key: str, default: T, cast: Callable[[Any], T]) -> T:
    """environment variable > configs/config.yaml (evaluation.<key>) > default. A bad value is logged and ignored."""
    raw: Any = os.getenv(env_name)
    source = f"environment variable {env_name}"
    if raw is None or str(raw).strip() == "":
        raw = get_config(f"evaluation.{key}", None)
        source = f"config key evaluation.{key}"
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("Ignoring invalid value %r from %s (%s); using %r", raw, source, exc, default)
        return default


@dataclass(frozen=True)
class EvaluationSettings:
    """Everything the service needs to know about its surroundings."""

    retrieval_url: str = "http://retrieval:8003"
    preprocessing_url: str = "http://preprocessing:8001"
    query_refinement_url: str = "http://query_refinement:8005"
    retrieval_timeout_seconds: float = 60.0
    preprocessing_timeout_seconds: float = 10.0
    refinement_timeout_seconds: float = 10.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    #: Queries evaluated at the same time. 1 keeps the old sequential behaviour.
    workers: int = 8
    default_top_k: int = 100
    max_top_k: int = 1000
    max_query_limit: int = 1000
    save_results: bool = True
    history_limit: int = 100

    @classmethod
    def load(cls) -> "EvaluationSettings":
        return cls(
            retrieval_url=_setting("EVALUATION_RETRIEVAL_URL", "retrieval_url", cls.retrieval_url, _as_url),
            preprocessing_url=_setting("EVALUATION_PREPROCESSING_URL", "preprocessing_url", cls.preprocessing_url, _as_url),
            query_refinement_url=_setting(
                "EVALUATION_QUERY_REFINEMENT_URL", "query_refinement_url", cls.query_refinement_url, _as_url
            ),
            retrieval_timeout_seconds=_setting(
                "EVALUATION_RETRIEVAL_TIMEOUT", "retrieval_timeout_seconds",
                cls.retrieval_timeout_seconds, _as_positive_float,
            ),
            preprocessing_timeout_seconds=_setting(
                "EVALUATION_PREPROCESSING_TIMEOUT", "preprocessing_timeout_seconds",
                cls.preprocessing_timeout_seconds, _as_positive_float,
            ),
            refinement_timeout_seconds=_setting(
                "EVALUATION_REFINEMENT_TIMEOUT", "refinement_timeout_seconds",
                cls.refinement_timeout_seconds, _as_positive_float,
            ),
            max_retries=max(1, _setting("EVALUATION_MAX_RETRIES", "max_retries", cls.max_retries, _as_positive_int)),
            retry_backoff_seconds=_setting(
                "EVALUATION_RETRY_BACKOFF", "retry_backoff_seconds",
                cls.retry_backoff_seconds, lambda value: max(0.0, float(value)),
            ),
            workers=min(MAX_WORKERS, max(
                1, _setting("EVALUATION_WORKERS", "workers", cls.workers, _as_positive_int)
            )),
            default_top_k=_setting("EVALUATION_DEFAULT_TOP_K", "default_top_k", cls.default_top_k, _as_positive_int),
            max_top_k=_setting("EVALUATION_MAX_TOP_K", "max_top_k", cls.max_top_k, _as_positive_int),
            max_query_limit=_setting(
                "EVALUATION_MAX_QUERY_LIMIT", "max_query_limit", cls.max_query_limit, _as_positive_int
            ),
            save_results=_setting("EVALUATION_SAVE_RESULTS", "save_results", cls.save_results, _as_bool),
            history_limit=_setting("EVALUATION_HISTORY_LIMIT", "history_limit", cls.history_limit, _as_positive_int),
        )

    def top_k(self, requested: int | None) -> int:
        """Clamp a requested cutoff into the supported range."""
        if requested is None:
            return self.default_top_k
        return max(1, min(int(requested), self.max_top_k))

    def query_limit(self, requested: int | None) -> int | None:
        """0/None means "every query"; anything larger is capped."""
        if not requested:
            return None
        return min(int(requested), self.max_query_limit)


@lru_cache(maxsize=1)
def get_settings() -> EvaluationSettings:
    return EvaluationSettings.load()


def reset_settings_cache() -> None:
    """For tests: forget the cached settings so the next get_settings() re-reads the environment."""
    get_settings.cache_clear()
