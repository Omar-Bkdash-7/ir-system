"""Deleting built indices (this logic used to be written inline inside main.py, with its own slug function)."""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from services.indexing.config import indices_base_path
from services.indexing.errors import InvalidDatasetIdError
from services.indexing.naming import index_slug, validate_index_type
from services.indexing.services.index_cache import clear_index_cache

logger = logging.getLogger(__name__)


def _dataset_index_dir(dataset_name: str) -> Path:
    base = indices_base_path()
    path = base / index_slug(dataset_name)  # the SAME slug the indexers use to write; rejects '', '.', '..'
    if path.resolve().parent != base.resolve():
        raise InvalidDatasetIdError(f"Invalid dataset name: {dataset_name!r}")
    return path


def delete_index(index_type: str, dataset_name: str) -> Dict[str, Any]:
    """Delete one index type of a dataset: data/indices/<slug>/<index_type>/."""
    validate_index_type(index_type)  # InvalidIndexTypeError: "Invalid index type: <x>"
    index_dir = _dataset_index_dir(dataset_name) / index_type
    deleted_files: List[str] = []
    if index_dir.is_dir():
        for file_path in index_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
                deleted_files.append(file_path.name)
                logger.info(f"Deleted: {file_path}")
        try:
            index_dir.rmdir()
            logger.info(f"Removed empty directory: {index_dir}")
        except OSError:
            pass  # directory not empty or permission issue
    clear_index_cache()
    if deleted_files:
        return {"status": "success", "message": f"Deleted {index_type} index",
                "deleted_files": deleted_files, "count": len(deleted_files)}
    return {"status": "not_found", "message": f"No {index_type} index files found",
            "deleted_files": [], "count": 0}


def delete_all_indices(dataset_name: str) -> Dict[str, Any]:
    """Delete every index of a dataset: data/indices/<slug>/."""
    dataset_dir = _dataset_index_dir(dataset_name)
    if dataset_dir.is_dir():
        file_count = sum(1 for p in dataset_dir.rglob("*") if p.is_file())
        shutil.rmtree(dataset_dir)
        clear_index_cache()
        logger.info(f"Deleted {file_count} index files for dataset: {dataset_name}")
        return {"status": "success", "message": f"Deleted all indices for {dataset_name}",
                "deleted_count": file_count}
    return {"status": "not_found", "message": "No index directory found", "deleted_count": 0}
