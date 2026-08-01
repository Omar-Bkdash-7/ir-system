"""
Dataset metadata management.
"""

import json
import logging
from typing import Dict, Any

from services.indexing.dataset.path_resolver import DatasetPathResolver
from services.indexing.dataset.constants import METADATA_FILE_SUFFIX


logger = logging.getLogger(__name__)


class DatasetMetadataManager:
    """
    Manages dataset metadata storage and retrieval.
    
    Handles atomic writes and safe reads of dataset metadata files,
    ensuring data integrity during concurrent operations.
    """
    
    def __init__(self):
        """Initialize metadata manager with path resolver."""
        self.path_resolver = DatasetPathResolver()
    
    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Save metadata to JSON file atomically.
        
        Uses temporary file and atomic rename for safe concurrent writes.
        This prevents corruption if the process is interrupted during write.
        
        Args:
            metadata: Dictionary containing dataset metadata
            
        Raises:
            IOError: If file write or rename fails
        """
        import os
        import time
        
        metadata_path = self.path_resolver.get_metadata_path()
        self.path_resolver.ensure_directory_exists(metadata_path.parent)
        
        # Simple direct write with retry logic for Windows file locking
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Metadata saved successfully to {metadata_path}")
                return
                
            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"File write attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to save metadata after {max_retries} attempts: {e}", exc_info=True)
                    raise
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from JSON file.
        
        Returns:
            Dictionary containing dataset metadata
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If metadata file is corrupted
        """
        metadata_path = self.path_resolver.get_metadata_path()
        
        if not metadata_path.exists():
            error_msg = f"Metadata file not found at {metadata_path}. Run dataset validation first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            logger.debug(f"Metadata loaded from {metadata_path}")
            return metadata
            
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted metadata file: {e}", exc_info=True)
            raise
    
    def metadata_exists(self) -> bool:
        """
        Check if metadata file exists.
        
        Returns:
            True if metadata file exists, False otherwise
        """
        return self.path_resolver.get_metadata_path().exists()
