"""Runtime settings of the retrieval service.

Every value has a default in code. It can be overridden in configs/config.yaml (section ``retrieval:``) and, for the
deployment-level knobs, by an environment variable (the variable wins). No configuration change is required to run.

    key (retrieval.<key>)          environment variable                 default
    -----------------------------  -----------------------------------  ------------------------
    compact_cache_dir              RETRIEVAL_CACHE_DIR                  data/cache/retrieval
    stale_check_seconds            RETRIEVAL_STALE_CHECK_SECONDS        2.0
    index_memory_budget_mb         RETRIEVAL_INDEX_MEMORY_MB            2048
    embedding_query_mode           RETRIEVAL_EMBEDDING_QUERY_MODE       raw
    embedding_mmap                 RETRIEVAL_EMBEDDING_MMAP             true
    query_cache_size               RETRIEVAL_QUERY_CACHE_SIZE           512
    db_pool_max                    RETRIEVAL_DB_POOL_MAX                8
    parallel_workers               RETRIEVAL_PARALLEL_WORKERS           8
    warmup                         RETRIEVAL_WARMUP                     (nothing: indexes load on first use)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Tuple, TypeVar

from common.config import get_config

logger = logging.getLogger(__name__)

#: "raw": the original text is embedded — the indexer embeds each document's RAW text, so this is the coherent
#: default; "preprocessed": the query text is tokenised/lemmatised before it is embedded (what the service did first).
QUERY_MODES = ("preprocessed", "raw")

T = TypeVar("T")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _as_query_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in QUERY_MODES:
        raise ValueError(f"must be one of {QUERY_MODES}")
    return mode


def _as_positive_int(value: Any) -> int:
    number = int(value)
    if number < 0:
        raise ValueError("must be >= 0")
    return number


def _as_warmup(value: Any) -> Tuple[str, ...]:
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    return tuple(item.strip() for item in items if str(item).strip())


def _setting(env_name: str, key: str, default: T, cast: Callable[[Any], T]) -> T:
    """environment variable > configs/config.yaml (retrieval.<key>) > default. A bad value is logged and ignored."""
    raw: Any = os.getenv(env_name)
    source = f"environment variable {env_name}"
    if raw is None or str(raw).strip() == "":
        raw = get_config(f"retrieval.{key}", None)
        source = f"config key retrieval.{key}"
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("Ignoring invalid value %r from %s (%s); using %r", raw, source, exc, default)
        return default


@dataclass(frozen=True)
class RetrievalSettings:
    cache_dir: Path = Path("data/cache/retrieval")
    stale_check_seconds: float = 2.0
    index_memory_budget_mb: int = 2048
    embedding_query_mode: str = "raw"
    embedding_mmap: bool = True
    query_cache_size: int = 512
    db_pool_max: int = 8
    parallel_workers: int = 8
    warmup: Tuple[str, ...] = ()
    bm25_default_k1: float = 1.5
    bm25_default_b: float = 0.75

    @classmethod
    def load(cls) -> "RetrievalSettings":
        return cls(
            cache_dir=_setting("RETRIEVAL_CACHE_DIR", "compact_cache_dir", cls.cache_dir, Path),
            stale_check_seconds=_setting("RETRIEVAL_STALE_CHECK_SECONDS", "stale_check_seconds", cls.stale_check_seconds, float),
            index_memory_budget_mb=_setting("RETRIEVAL_INDEX_MEMORY_MB", "index_memory_budget_mb", cls.index_memory_budget_mb, _as_positive_int),
            embedding_query_mode=_setting("RETRIEVAL_EMBEDDING_QUERY_MODE", "embedding_query_mode", cls.embedding_query_mode, _as_query_mode),
            embedding_mmap=_setting("RETRIEVAL_EMBEDDING_MMAP", "embedding_mmap", cls.embedding_mmap, _as_bool),
            query_cache_size=_setting("RETRIEVAL_QUERY_CACHE_SIZE", "query_cache_size", cls.query_cache_size, _as_positive_int),
            db_pool_max=max(1, _setting("RETRIEVAL_DB_POOL_MAX", "db_pool_max", cls.db_pool_max, _as_positive_int)),
            parallel_workers=max(1, _setting("RETRIEVAL_PARALLEL_WORKERS", "parallel_workers", cls.parallel_workers, _as_positive_int)),
            warmup=_setting("RETRIEVAL_WARMUP", "warmup", cls.warmup, _as_warmup),
            bm25_default_k1=float(get_config("models.bm25.default_k1", cls.bm25_default_k1)),
            bm25_default_b=float(get_config("models.bm25.default_b", cls.bm25_default_b)),
        )


@lru_cache(maxsize=1)
def get_settings() -> RetrievalSettings:
    return RetrievalSettings.load()


def reset_settings_cache() -> None:
    """For tests: forget the cached settings so the next get_settings() re-reads the environment."""
    get_settings.cache_clear()
