"""
Dataset Management Package

Modular dataset management service for the IR system.
Provides validation, loading, scanning, and database operations
for IR datasets.

Architecture:
    - service: Main orchestration service
    - validator: Dataset validation logic
    - loader: Dataset loading with retry
    - metadata_manager: Metadata persistence
    - scanner: Local dataset discovery
    - deleter: Safe dataset deletion
    - db_operations: PostgreSQL integration
    - path_resolver: Path management
    - constants: Configuration constants

Public API:
    - validate_and_store_metadata: Validate and store dataset metadata
    - get_metadata: Retrieve stored metadata
    - list_local_datasets: List downloaded datasets
    - delete_local_dataset: Delete local dataset
    - upload_dataset_to_db: Upload to PostgreSQL
    - get_db_dataset_info: Get database statistics
"""

import logging
import os
from pathlib import Path

from services.indexing._lazy import install_lazy_exports

_logger = logging.getLogger(__name__)


def _ensure_ir_datasets_home() -> None:
    """
    Create ``IR_DATASETS_HOME`` before anything imports ``ir_datasets``.

    ``ir_datasets`` creates its download root while being imported, so a missing or
    unwritable directory raises ``PermissionError`` at import time and takes down every
    endpoint of the modules that import it -- including the ones that only read the
    filesystem, such as listing local datasets or metadata.

    Creating the root here, once per package import, keeps that failure out of the import
    chain. A directory this process is not allowed to create is left alone: the calls that
    genuinely need ``ir_datasets`` then fail individually with a clear message.
    """
    home = os.environ.get("IR_DATASETS_HOME")
    if not home:
        return
    try:
        Path(home).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _logger.warning("Cannot create IR_DATASETS_HOME %s: %s", home, exc)


_ensure_ir_datasets_home()

# Public names are resolved on first access (PEP 562), so importing this package no longer drags in psycopg2,
# ir_datasets, scikit-learn and numpy. Every old import path keeps working.
install_lazy_exports(globals(), __name__, {
    "validate_and_store_metadata": "services.indexing.dataset.api",
    "get_metadata": "services.indexing.dataset.api",
    "list_local_datasets": "services.indexing.dataset.api",
    "delete_local_dataset": "services.indexing.dataset.api",
    "upload_dataset_to_db": "services.indexing.dataset.api",
    "delete_dataset_from_db": "services.indexing.dataset.api",
    "get_db_dataset_info": "services.indexing.dataset.api",
    "DatasetService": "services.indexing.dataset.service",
    "DatasetValidator": "services.indexing.dataset.validator",
    "DatasetLoader": "services.indexing.dataset.loader",
    "DatasetMetadataManager": "services.indexing.dataset.metadata_manager",
    "LocalDatasetScanner": "services.indexing.dataset.scanner",
    "LocalDatasetDeleter": "services.indexing.dataset.deleter",
    "DatasetUploader": "services.indexing.dataset.db_operations",
    "DatabaseInfoRetriever": "services.indexing.dataset.db_operations",
    "DatasetDatabaseDeleter": "services.indexing.dataset.db_operations",
    "database_cursor": "services.indexing.dataset.db_operations",
    "DatasetPathResolver": "services.indexing.dataset.path_resolver",
    "DEFAULT_BATCH_SIZE": "services.indexing.dataset.constants",
    "DEFAULT_VALIDATION_RETRIES": "services.indexing.dataset.constants",
    "DEFAULT_RETRY_DELAY": "services.indexing.dataset.constants",
    "DATA_FILE_EXTENSIONS": "services.indexing.dataset.constants",
})

__all__ = [
    # Public API functions
    "validate_and_store_metadata",
    "get_metadata",
    "list_local_datasets",
    "delete_local_dataset",
    "upload_dataset_to_db",
    "delete_dataset_from_db",
    "get_db_dataset_info",
    
    # Service classes
    "DatasetService",
    "DatasetValidator",
    "DatasetLoader",
    "DatasetMetadataManager",
    "LocalDatasetScanner",
    "LocalDatasetDeleter",
    "DatasetUploader",
    "DatasetDatabaseDeleter",
    "DatabaseInfoRetriever",
    
    # Utilities
    "DatasetPathResolver",
    "database_cursor",
    
    # Constants
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_VALIDATION_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "DATA_FILE_EXTENSIONS",
]

__version__ = "1.0.0"
