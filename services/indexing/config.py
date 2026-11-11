"""Settings of the indexing service.

Every value has a default in code. It can also come from the shared
``configs/config.yaml`` (the key shown below) and, for the deployment-level knobs,
from an environment variable, which wins. Nothing has to be configured to run.

    environment variable               config key                    default
    ---------------------------------  ----------------------------  ---------------
    INDEXING_INDICES_DIR               indices.base_path             data/indices
    INDEXING_TABLE_PREFIX              database.table_prefix         dataset
    INDEXING_PROCESSED_DIR             paths.processed               data/processed
    INDEXING_SERVER_CURSOR_ROWS        gateway-ish tuning           2000
    INDEXING_UPLOAD_BATCH_SIZE         database.batch_size           1000
    INDEXING_VALIDATION_RETRIES        dataset.validation_retries    3
    INDEXING_VALIDATION_RETRY_DELAY    dataset.validation_retry_delay 2
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypeVar

from common.config import get_config
from services.indexing.dataset.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RETRY_DELAY,
    DEFAULT_VALIDATION_RETRIES,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _as_positive_int(value: Any) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError("must be > 0")
    return number


def _as_non_negative_float(value: Any) -> float:
    number = float(value)
    if number < 0:
        raise ValueError("must be >= 0")
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
class IndexingSettings:
    """Where indices and metadata live, and how the build reads documents."""

    indices_dir: str = "data/indices"
    table_prefix: str = "dataset"
    processed_dir: str = "data/processed"
    #: rows fetched per round trip while streaming a dataset out of PostgreSQL
    server_cursor_rows: int = 2000
    upload_batch_size: int = DEFAULT_BATCH_SIZE
    validation_retries: int = DEFAULT_VALIDATION_RETRIES
    validation_retry_delay: float = DEFAULT_RETRY_DELAY

    @classmethod
    def load(cls) -> "IndexingSettings":
        return cls(
            indices_dir=_setting("INDEXING_INDICES_DIR", "indices.base_path", cls.indices_dir, str),
            table_prefix=_setting("INDEXING_TABLE_PREFIX", "database.table_prefix", cls.table_prefix, str),
            processed_dir=_setting("INDEXING_PROCESSED_DIR", "paths.processed", cls.processed_dir, str),
            server_cursor_rows=_setting("INDEXING_SERVER_CURSOR_ROWS", "indexing.server_cursor_rows",
                                        cls.server_cursor_rows, _as_positive_int),
            upload_batch_size=_setting("INDEXING_UPLOAD_BATCH_SIZE", "database.batch_size",
                                       cls.upload_batch_size, _as_positive_int),
            validation_retries=_setting("INDEXING_VALIDATION_RETRIES", "dataset.validation_retries",
                                        cls.validation_retries, _as_positive_int),
            validation_retry_delay=_setting("INDEXING_VALIDATION_RETRY_DELAY", "dataset.validation_retry_delay",
                                            cls.validation_retry_delay, _as_non_negative_float),
        )


@lru_cache(maxsize=1)
def get_settings() -> IndexingSettings:
    return IndexingSettings.load()


def reset_settings_cache() -> None:
    """For tests: forget the cached settings so the next get_settings() re-reads the environment."""
    get_settings.cache_clear()


def indices_base_path() -> Path:
    """Root directory of all built indices."""
    return Path(get_settings().indices_dir)


def table_prefix() -> str:
    """Prefix of every dataset's PostgreSQL table (the naming rules are in `naming.table_name`)."""
    return get_settings().table_prefix
