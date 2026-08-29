"""
Main dataset service orchestration.

Supports multi-dataset validation and metadata management.
"""

import logging
from typing import Dict, Any, Optional

from common.config import get_config
from services.indexing.dataset.constants import (
    DEFAULT_DATASET_ID,
    DEFAULT_VALIDATION_RETRIES,
    DEFAULT_RETRY_DELAY
)
from services.indexing.dataset.validator import DatasetValidator
from services.indexing.dataset.loader import DatasetLoader
from services.indexing.dataset.metadata_manager import DatasetMetadataManager


logger = logging.getLogger(__name__)


class DatasetService:
    """
    Main service for dataset validation and management operations.
    
    Orchestrates validation, loading, and metadata management workflows.
    Supports multiple datasets with individual metadata storage.
    """
    
    def __init__(self):
        """Initialize dataset service with required managers."""
        self.metadata_manager = DatasetMetadataManager()
    
    def validate_and_store_metadata(
        self, 
        dataset_id: Optional[str] = None,
        skip_min_thresholds: bool = False,
        set_as_active: bool = True
    ) -> Dict[str, Any]:
        """
        Validate dataset and store its metadata.
        
        Performs complete validation workflow:
        1. Check dataset exists in registry
        2. Verify required features (queries, qrels)
        3. Load dataset with retry logic
        4. Extract and store metadata
        5. Optionally set as active dataset
        
        Args:
            dataset_id: IR datasets identifier (e.g., 'beir/quora')
            skip_min_thresholds: Reserved for future threshold validation
            set_as_active: Whether to set this dataset as active (default: True)
            
        Returns:
            Dictionary containing dataset metadata and validation status
            
        Raises:
            DatasetValidationError: If validation fails
        """
        dataset_id = dataset_id or get_config("dataset.ir_datasets_id", DEFAULT_DATASET_ID)
        
        logger.info(f"Starting validation for dataset: {dataset_id}")
        
        # Step 1: Validate existence
        validator = DatasetValidator(dataset_id)
        validator.validate_exists()
        
        # Step 2: Validate features
        validator.validate_has_required_features()
        
        # Step 3: Load with retry
        max_retries = get_config("dataset.validation_retries", DEFAULT_VALIDATION_RETRIES)
        retry_delay = get_config("dataset.validation_retry_delay", DEFAULT_RETRY_DELAY)
        
        loader = DatasetLoader(dataset_id)
        metadata = loader.load_with_retry(max_retries, retry_delay)
        
        # Step 4: Store metadata (also sets as active if requested)
        self.metadata_manager.save_metadata(metadata, dataset_name=dataset_id)
        
        logger.info(f"Validation completed successfully for '{dataset_id}'")
        logger.info(f"Active dataset: {self.metadata_manager.get_active_dataset()}")
        
        return metadata
    
    def get_metadata(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve stored dataset metadata.
        
        Args:
            dataset_name: Dataset identifier. If None, returns active dataset
        
        Returns:
            Dictionary containing dataset metadata
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        return self.metadata_manager.load_metadata(dataset_name)
    
    def has_metadata(self, dataset_name: Optional[str] = None) -> bool:
        """
        Check if metadata exists for a dataset.
        
        Args:
            dataset_name: Dataset identifier. If None, checks main metadata
        
        Returns:
            True if metadata file exists, False otherwise
        """
        return self.metadata_manager.metadata_exists(dataset_name)
    
    def get_active_dataset(self) -> Optional[str]:
        """
        Get the currently active dataset name.
        
        Returns:
            Active dataset name or None
        """
        return self.metadata_manager.get_active_dataset()
    
    def set_active_dataset(self, dataset_name: str) -> None:
        """
        Set the active dataset.
        
        Args:
            dataset_name: Dataset identifier to set as active
        """
        self.metadata_manager.set_active_dataset(dataset_name)
        logger.info(f"Active dataset changed to: {dataset_name}")
    
    def list_datasets(self) -> list:
        """
        List all datasets with metadata.
        
        Returns:
            List of dataset metadata dictionaries
        """
        return self.metadata_manager.list_datasets()
