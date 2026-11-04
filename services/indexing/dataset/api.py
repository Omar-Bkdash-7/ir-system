"""
Public API functions of the dataset package.

They used to be defined inside dataset/__init__.py, which therefore had to import every class (psycopg2,
ir_datasets, ...) just to be imported. The package now exposes them lazily from here; nothing else changed.
"""

from services.indexing.dataset.service import DatasetService
from services.indexing.dataset.scanner import LocalDatasetScanner
from services.indexing.dataset.deleter import LocalDatasetDeleter
from services.indexing.dataset.db_operations import (
    DatasetUploader,
    DatabaseInfoRetriever,
    DatasetDatabaseDeleter,
)
from services.indexing.dataset.constants import DEFAULT_BATCH_SIZE


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


def get_metadata(dataset_name=None):
    """
    Public API: Retrieve stored dataset metadata.
    
    Args:
        dataset_name: Dataset identifier. If None, returns active dataset
    
    Returns:
        Dictionary containing dataset metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    service = DatasetService()
    return service.get_metadata(dataset_name)


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
