"""Dataset operations the HTTP layer exposes: validate, list, download, upload, delete.

The implementations (ir_datasets loading, local scanning, PostgreSQL bulk copy) live
in `services/indexing/dataset/`; they are imported lazily so that importing the
service's core does not pull ir_datasets or psycopg2 in.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from services.indexing.config import get_settings

logger = logging.getLogger(__name__)


class DatasetOperations:
    """Thin, testable seam over the dataset package."""

    def __init__(self, batch_size: Optional[int] = None) -> None:
        self._batch_size = batch_size or get_settings().upload_batch_size

    def validate(self, dataset_id: str) -> Dict[str, Any]:
        """Validate a dataset with ir_datasets and store its metadata."""
        from services.indexing.dataset import validate_and_store_metadata

        return validate_and_store_metadata(dataset_id)

    def download(self, dataset_id: str) -> Dict[str, Any]:
        """Force download/verify of a dataset."""
        from services.indexing.dataset import maintenance

        return maintenance.download_dataset(dataset_id)

    def fix_corrupted(self, dataset_id: str) -> Dict[str, Any]:
        """Delete the cached files of a dataset and fetch them again (the id is checked before anything is removed)."""
        from services.indexing.dataset import maintenance

        return maintenance.fix_corrupted_dataset(dataset_id)

    def list_local(self) -> List[Dict[str, Any]]:
        """Datasets already present in IR_DATASETS_HOME."""
        from services.indexing.dataset import list_local_datasets

        datasets = list_local_datasets()
        logger.info("Found %d local datasets", len(datasets))
        return datasets

    def delete_local(self, dataset_id: str) -> Dict[str, Any]:
        """Remove a local dataset directory."""
        from services.indexing.dataset import delete_local_dataset

        return delete_local_dataset(dataset_id)

    def upload(self, dataset_id: str, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Copy a local dataset's documents into PostgreSQL, in batches."""
        from services.indexing.dataset import upload_dataset_to_db

        return upload_dataset_to_db(dataset_id, batch_size=batch_size or self._batch_size)

    def delete_from_database(self, dataset_id: str) -> Dict[str, Any]:
        """Drop the documents of one dataset from PostgreSQL."""
        from services.indexing.dataset import delete_dataset_from_db

        return delete_dataset_from_db(dataset_id)

    def database_info(self) -> Dict[str, Any]:
        """Tables and row counts of the documents database."""
        from services.indexing.dataset import get_db_dataset_info

        return get_db_dataset_info()

    def metadata(self) -> List[Dict[str, Any]]:
        """Stored metadata of every validated dataset."""
        from services.indexing.dataset.service import DatasetService

        return DatasetService().list_datasets()
