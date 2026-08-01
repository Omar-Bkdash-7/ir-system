"""
Local dataset deletion.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict

from services.indexing.dataset.path_resolver import DatasetPathResolver


logger = logging.getLogger(__name__)


class LocalDatasetDeleter:
    """
    Deletes local dataset directories.
    
    Provides safe deletion of dataset directories with validation
    and logging for audit trails.
    """
    
    def __init__(self):
        """Initialize deleter with path resolver."""
        self.path_resolver = DatasetPathResolver()
    
    def delete_dataset(self, dataset_id: str) -> Dict[str, str]:
        """
        Delete a local dataset directory.
        
        Performs validation before deletion to ensure safe operation.
        
        Args:
            dataset_id: Dataset identifier (e.g., 'beir/quora')
            
        Returns:
            Dictionary with deletion confirmation
            
        Raises:
            FileNotFoundError: If dataset directory doesn't exist
            ValueError: If dataset_id is invalid
        """
        if not dataset_id or not dataset_id.strip():
            raise ValueError("Dataset ID cannot be empty")
        
        target_path = self._resolve_dataset_path(dataset_id)
        
        if not target_path.exists():
            error_msg = f"Dataset directory not found: {target_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not target_path.is_dir():
            raise ValueError(f"Target path is not a directory: {target_path}")
        
        logger.info(f"Deleting dataset directory: {target_path}")
        
        try:
            shutil.rmtree(target_path)
            logger.info(f"Successfully deleted dataset '{dataset_id}' at {target_path}")
            
            return {
                "deleted": dataset_id,
                "path": str(target_path),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Failed to delete dataset '{dataset_id}': {e}", exc_info=True)
            raise
    
    def _resolve_dataset_path(self, dataset_id: str) -> Path:
        """
        Resolve dataset ID to filesystem path.
        
        Args:
            dataset_id: Dataset identifier with '/' separators
            
        Returns:
            Resolved Path object
        """
        home = self.path_resolver.get_ir_home()
        parts = dataset_id.strip("/").split("/")
        
        target = home
        for part in parts:
            target = target / part
        
        return target
