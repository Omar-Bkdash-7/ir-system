"""
Main dataset service orchestration.
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
    
    Orchestrates validation, loading, and metadata management workflows
    for the IR system dataset management service.
    """
    
    def __init__(self):
        """Initialize dataset service with required managers."""
        self.metadata_manager = DatasetMetadataManager()
    
    def validate_and_store_metadata(
        self, 
        dataset_id: Optional[str] = None,
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
            dataset_id: IR datasets identifier (e.g., 'beir/quora')
            skip_min_thresholds: Reserved for future threshold validation
            
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
        
        # Step 4: Store metadata
        self.metadata_manager.save_metadata(metadata)
        
        logger.info(f"Validation completed successfully for '{dataset_id}'")
        return metadata
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve stored dataset metadata.
        
        Returns:
            Dictionary containing dataset metadata
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        return self.metadata_manager.load_metadata()
    
    def has_metadata(self) -> bool:
        """
        Check if metadata exists.
        
        Returns:
            True if metadata file exists, False otherwise
        """
        return self.metadata_manager.metadata_exists()
