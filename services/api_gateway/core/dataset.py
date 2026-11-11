"""Reading saved dataset metadata from disk.

The gateway answers ``/datasets/info`` from the metadata files the indexing
service writes, and reads a dataset's preprocessing config from the same place
before tokenising a query. Both used to be spread over two client modules (one of
which imported `services.indexing`, pulling `ir_datasets` into every request).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Answer when nothing has been validated yet, so the frontend can render its empty state.
FALLBACK_INFO: Dict[str, Any] = {
    "dataset_name": None,
    "doc_count": 0,
    "query_count": 0,
    "qrels_count": 0,
    "validation_status": "not_validated",
    "source": "fallback",
    "note": "No datasets validated. Please validate a dataset first.",
}


def fallback_info() -> Dict[str, Any]:
    return dict(FALLBACK_INFO)


def metadata_file(metadata_dir: str, dataset: str) -> Path:
    """File name of a dataset's metadata, exactly as the indexing service builds it."""
    stem = dataset.replace("/", "-").replace("\\", "-")
    return Path(metadata_dir) / f"{stem}.json"


def latest_metadata_file(metadata_dir: str) -> Optional[Path]:
    """The most recently written metadata file (ties broken by name, so the answer is stable)."""
    directory = Path(metadata_dir)
    if not directory.is_dir():
        return None
    files = [path for path in directory.glob("*.json") if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: (path.stat().st_mtime, path.name))


def dataset_info(metadata_dir: str) -> Dict[str, Any]:
    """Metadata of the newest validated dataset, or the documented fallback."""
    path = latest_metadata_file(metadata_dir)
    if path is None:
        logger.warning("No dataset metadata in %s; answering with the fallback", metadata_dir)
        return fallback_info()

    try:
        with open(path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not read metadata %s: %s; answering with the fallback", path, exc)
        return fallback_info()

    logger.debug("Dataset info from %s: %s", path.name, metadata.get("dataset_name"))
    return metadata


def preprocessing_config(metadata_dir: str, dataset: Optional[str]) -> Optional[Dict[str, Any]]:
    """Saved preprocessing config of ``dataset``, or None when it was never saved."""
    if not dataset:
        return None

    path = metadata_file(metadata_dir, dataset)
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle).get("preprocessing_config")
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read the preprocessing config of %s: %s", dataset, exc)
        return None

    if config:
        logger.debug("Using the saved preprocessing config of %s", dataset)
    return config
