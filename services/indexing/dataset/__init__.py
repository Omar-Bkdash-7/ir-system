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

from services.indexing.dataset.service import DatasetService
from services.indexing.dataset.validator import DatasetValidator
from services.indexing.dataset.loader import DatasetLoader
from services.indexing.dataset.metadata_manager import DatasetMetadataManager
from services.indexing.dataset.scanner import LocalDatasetScanner
from services.indexing.dataset.deleter import LocalDatasetDeleter
from services.indexing.dataset.db_operations import (
    DatasetUploader,
    DatabaseInfoRetriever,
    DatasetDatabaseDeleter,
    database_cursor
)
from services.indexing.dataset.path_resolver import DatasetPathResolver
from services.indexing.dataset.constants import (
    DEFAULT_DATASET_ID,
    DEFAULT_BATCH_SIZE,
    DEFAULT_VALIDATION_RETRIES,
    DEFAULT_RETRY_DELAY,
    DATA_FILE_EXTENSIONS
)


# Public API Functions

def validate_and_store_metadata(dataset_id=None, skip_min_thresholds=False):
    """
    Public API: Validate dataset and store its metadata.
    
    Entry point for dataset validation workflow in the indexing service.
    
    Args:
        dataset_id: IR datasets identifier
        skip_min_thresholds: Reserved for future threshold validation
        
    Returns:
        Dictionary containing dataset metadata
        
    Raises:
        DatasetValidationError: If validation fails
    """
    service = DatasetService()
    return service.validate_and_store_metadata(dataset_id, skip_min_thresholds)


def get_metadata():
    """
    Public API: Retrieve stored dataset metadata.
    
    Returns:
        Dictionary containing dataset metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    service = DatasetService()
    return service.get_metadata()


def list_local_datasets():
    """
    Public API: List all locally downloaded datasets.
    
    Returns:
        List of dictionaries containing dataset information
    """
    scanner = LocalDatasetScanner()
    return scanner.list_datasets()


def delete_local_dataset(dataset_id):
    """
    Public API: Delete a local dataset directory.
    
    Args:
        dataset_id: Dataset identifier to delete
        
    Returns:
        Dictionary with deletion confirmation
        
    Raises:
        FileNotFoundError: If dataset doesn't exist
        ValueError: If dataset_id is invalid
    """
    deleter = LocalDatasetDeleter()
    return deleter.delete_dataset(dataset_id)


def upload_dataset_to_db(dataset_id, batch_size=DEFAULT_BATCH_SIZE):
    """
    Public API: Upload dataset documents to PostgreSQL.
    
    Entry point for dataset upload workflow in the indexing service.
    
    Args:
        dataset_id: IR datasets identifier
        batch_size: Number of documents per batch
        
    Returns:
        Dictionary with upload statistics
        
    Raises:
        ValueError: If dataset is invalid
        psycopg2.Error: If database operation fails
    """
    uploader = DatasetUploader()
    return uploader.upload_to_db(dataset_id, batch_size)


def delete_dataset_from_db(dataset_id: str):
    """
    Public API: Delete dataset documents from PostgreSQL.
    
    Args:
        dataset_id: Dataset identifier to delete
        
    Returns:
        Dictionary with deletion statistics
        
    Raises:
        psycopg2.Error: If database operation fails
    """
    deleter = DatasetDatabaseDeleter()
    return deleter.delete_dataset_from_db(dataset_id)


def get_db_dataset_info():
    """
    Public API: Get database table information.
    
    Returns:
        Dictionary with table statistics and health information
    """
    retriever = DatabaseInfoRetriever()
    return retriever.get_info()


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
    "DEFAULT_DATASET_ID",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_VALIDATION_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "DATA_FILE_EXTENSIONS",
]

__version__ = "1.0.0"
