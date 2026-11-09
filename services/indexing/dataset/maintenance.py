"""Dataset maintenance operations that used to be written inline inside main.py."""
import logging
import shutil
from typing import Any, Dict

import ir_datasets

from services.indexing.dataset.path_resolver import DatasetPathResolver
from services.indexing.naming import safe_child_path

logger = logging.getLogger(__name__)

_SAMPLE_DOCS = 11  # documents read to force the download (the whole source file is fetched on first access)


def download_dataset(dataset_id: str) -> Dict[str, Any]:
    """Force download/verify of a dataset with ir_datasets (first documents sampled, queries and qrels counted)."""
    logger.info(f"Attempting to load dataset: {dataset_id}")
    ds = ir_datasets.load(dataset_id)  # downloads the dataset if it is not present
    doc_count = query_count = qrel_count = 0
    if hasattr(ds, "docs_iter"):
        for i, _ in enumerate(ds.docs_iter()):
            doc_count += 1
            if i >= _SAMPLE_DOCS - 1:
                break
    if hasattr(ds, "queries_iter"):
        query_count = sum(1 for _ in ds.queries_iter())
    if hasattr(ds, "qrels_iter"):
        qrel_count = sum(1 for _ in ds.qrels_iter())
    logger.info(f"Dataset {dataset_id} loaded: docs_sampled={doc_count}, queries={query_count}, qrels={qrel_count}")
    return {"status": "success", "dataset_id": dataset_id,
            "message": "Dataset downloaded and verified",
            "sample_counts": {"docs_sampled": doc_count, "queries": query_count, "qrels": qrel_count}}


def fix_corrupted_dataset(dataset_id: str) -> Dict[str, Any]:
    """Delete the cached files of one dataset and download it again.

    Everything is checked BEFORE anything is deleted: the id must be a non-empty relative id that stays inside the
    ir_datasets directory, and ir_datasets must know it. (An empty or '..' id used to wipe the whole cache, and the
    original handler also referred to `ir_datasets` without importing it, so it deleted the files and then failed.)"""
    logger.info(f"Attempting to fix corrupted dataset: {dataset_id}")
    dataset_path = safe_child_path(DatasetPathResolver().get_ir_home(), dataset_id)
    ir_datasets.load(dataset_id)  # unknown id: raises here, nothing has been touched yet
    if dataset_path.exists():
        logger.info(f"Deleting corrupted dataset at: {dataset_path}")
        shutil.rmtree(dataset_path)
        logger.info("Deleted corrupted dataset files")
    logger.info(f"Re-downloading dataset '{dataset_id}'...")
    ds = ir_datasets.load(dataset_id)
    doc_count = query_count = qrel_count = 0
    if hasattr(ds, "docs_iter"):
        logger.info("Iterating documents to force complete download...")
        for _ in ds.docs_iter():
            doc_count += 1
            if doc_count % 5000 == 0:
                logger.info(f"Downloaded {doc_count:,} documents...")
    if hasattr(ds, "queries_iter"):
        query_count = sum(1 for _ in ds.queries_iter())
    if hasattr(ds, "qrels_iter"):
        qrel_count = sum(1 for _ in ds.qrels_iter())
    logger.info(f"Dataset fixed successfully: docs={doc_count}, queries={query_count}, qrels={qrel_count}")
    return {"status": "success", "dataset_id": dataset_id,
            "message": "Dataset fixed and re-downloaded successfully",
            "counts": {"documents": doc_count, "queries": query_count, "qrels": qrel_count}}
