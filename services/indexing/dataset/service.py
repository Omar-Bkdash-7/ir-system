"""
Main dataset service orchestration.

Supports multi-dataset validation and metadata management.
"""

import logging
from typing import Dict, Any, Optional

from common.config import get_config
from services.indexing.dataset.constants import (
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
        dataset_id: str,
        skip_min_thresholds: bool = False
    ) -> Dict[str, Any]:
        """
        Validate dataset and store its metadata.
        
        Performs complete validation workflow:
        1. Check dataset exists in registry
        2. Verify required features (queries, qrels)
        3. Load dataset with retry logic
        4. Extract and store metadata
        
        Args:
            dataset_id: IR datasets identifier (e.g., 'beir/quora') - REQUIRED
            skip_min_thresholds: Reserved for future threshold validation
            
        Returns:
            Dictionary containing dataset metadata and validation status
            
        Raises:
            DatasetValidationError: If validation fails
            ValueError: If dataset_id not provided
        """
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
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
        
        # Step 4: Store metadata
        self.metadata_manager.save_metadata(metadata, dataset_name=dataset_id)
        
        logger.info(f"Validation completed successfully for '{dataset_id}'")
        
        return metadata
    
    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """
        Retrieve stored dataset metadata.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
        
        Returns:
            Dictionary containing dataset metadata
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            ValueError: If dataset_name not provided
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        return self.metadata_manager.load_metadata(dataset_name)
    
    def has_metadata(self, dataset_name: str) -> bool:
        """
        Check if metadata exists for a dataset.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
        
        Returns:
            True if metadata file exists, False otherwise
            
        Raises:
            ValueError: If dataset_name not provided
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        return self.metadata_manager.metadata_exists(dataset_name)
    
    def list_datasets(self) -> list:
        """
        List all datasets with metadata.
        
        Returns:
            List of dataset metadata dictionaries
        """
        return self.metadata_manager.list_datasets()
    
    def save_preprocessing_config(self, dataset_name: str, preprocessing_config: Optional[dict]) -> None:
        """
        Save preprocessing configuration for a dataset.

        The file is created when the dataset has no metadata yet (a dataset that was uploaded to PostgreSQL but
        never validated has none). Without this, the config of an index build was silently dropped and queries were
        preprocessed with the defaults — i.e. differently than the documents it is supposed to match.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
            preprocessing_config: Preprocessing configuration to save
            
        Raises:
            ValueError: If dataset_name not provided
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        try:
            metadata = self.metadata_manager.load_metadata(dataset_name)
        except FileNotFoundError:
            logger.info(f"No metadata for '{dataset_name}' yet: creating it to store the preprocessing config")
            metadata = {"dataset_name": dataset_name}
        
        # Update preprocessing config
        metadata["preprocessing_config"] = preprocessing_config
        
        # Save updated metadata
        self.metadata_manager.save_metadata(metadata, dataset_name=dataset_name)
        
        logger.info(f"Saved preprocessing config for dataset: {dataset_name}")
    
    def get_preprocessing_config(self, dataset_name: str) -> Optional[dict]:
        """
        Retrieve preprocessing configuration for a dataset.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
        
        Returns:
            Preprocessing configuration dict or None if not set
            
        Raises:
            ValueError: If dataset_name not provided
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        try:
            metadata = self.metadata_manager.load_metadata(dataset_name)
            return metadata.get("preprocessing_config")
        except FileNotFoundError:
            logger.warning(f"No metadata found for dataset: {dataset_name}")
            return None
