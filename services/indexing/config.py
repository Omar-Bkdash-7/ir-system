"""Configuration lookups shared by the indexing modules (one place instead of repeated magic strings)."""
from pathlib import Path

from common.config import get_config


def indices_base_path() -> Path:
    """Root directory of all built indices: `indices.base_path`, default data/indices.

    (config.yaml also has `indices.base_dir`, but nothing ever read it and the YAML path templates hard-code
    data/indices, so the key is intentionally not used here.)"""
    return Path(get_config("indices.base_path", "data/indices"))


def table_prefix() -> str:
    """Prefix of every dataset's PostgreSQL table: `database.table_prefix`, default 'dataset'."""
    return get_config("database.table_prefix", "dataset")
