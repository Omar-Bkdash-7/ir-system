"""
Dataset Service - Legacy Compatibility Wrapper

This module provides backward compatibility by re-exporting
the modular dataset management components.

For new code, import directly from services.indexing.dataset package.

Deprecated: This module will be removed in version 2.0.
Use: from services.indexing.dataset import validate_and_store_metadata
"""

import warnings

# Re-export all public APIs from the modular dataset package
from services.indexing.dataset import (
    # Public API functions
    validate_and_store_metadata,
    get_metadata,
    list_local_datasets,
    delete_local_dataset,
    upload_dataset_to_db,
    delete_dataset_from_db,
    get_db_dataset_info,
    
    # Service classes
    DatasetService,
    DatasetValidator,
    DatasetLoader,
    DatasetMetadataManager,
    LocalDatasetScanner,
    LocalDatasetDeleter,
    DatasetUploader,
    DatasetDatabaseDeleter,
    DatabaseInfoRetriever,
    
    # Utilities
    DatasetPathResolver,
    database_cursor,
)


# Deprecation warning
warnings.warn(
    "Direct import from dataset_service module is deprecated. "
    "Use 'from services.indexing.dataset import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)


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
]
