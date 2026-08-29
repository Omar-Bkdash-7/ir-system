"""
Dataset metadata management.

Supports multi-dataset metadata storage with an active dataset concept.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from services.indexing.dataset.path_resolver import DatasetPathResolver


logger = logging.getLogger(__name__)


class DatasetMetadataManager:
    """
    Manages multi-dataset metadata storage and retrieval.
    
    Features:
    - Supports multiple datasets with individual metadata files
    - Maintains an active dataset pointer
    - Atomic writes with retry logic for Windows compatibility
    - Backward compatible with legacy single metadata.json format
    
    Structure:
        data/
            metadata.json           # Main file with active dataset + all datasets info
            metadata/               # Individual dataset metadata files
                beir-quora.json
                cranfield.json
    """
    
    def __init__(self):
        """Initialize metadata manager with path resolver."""
        self.path_resolver = DatasetPathResolver()
        self.metadata_dir = self.path_resolver.get_metadata_path().parent / "metadata"
    
    def save_metadata(self, metadata: Dict[str, Any], dataset_name: Optional[str] = None) -> None:
        """
        Save metadata for a specific dataset and update main metadata file.
        
        Args:
            metadata: Dictionary containing dataset metadata
            dataset_name: Dataset identifier (e.g., 'beir/quora'). 
                         If None, uses metadata['dataset_name']
            
        Raises:
            IOError: If file write fails
            ValueError: If dataset_name not provided and not in metadata
        """
        # Extract dataset name
        dataset_name = dataset_name or metadata.get("dataset_name")
        if not dataset_name:
            raise ValueError("dataset_name must be provided either as argument or in metadata")
        
        # Ensure metadata contains dataset_name
        metadata["dataset_name"] = dataset_name
        
        # Save individual dataset metadata
        self._save_dataset_metadata(dataset_name, metadata)
        
        # Update main metadata file
        self._update_main_metadata(dataset_name, metadata)
        
        logger.info(f"Metadata saved for dataset: {dataset_name}")
    
    def load_metadata(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load metadata for a specific dataset or active dataset.
        
        Args:
            dataset_name: Dataset identifier. If None, loads active dataset
            
        Returns:
            Dictionary containing dataset metadata
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If metadata file is corrupted
        """
        if dataset_name is None:
            # Load active dataset
            main_metadata = self._load_main_metadata()
            dataset_name = main_metadata.get("active_dataset")
            
            if not dataset_name:
                raise FileNotFoundError("No active dataset set. Run dataset validation first.")
        
        # Try to load from individual file first
        dataset_metadata = self._load_dataset_metadata(dataset_name)
        
        if dataset_metadata:
            return dataset_metadata
        
        # Fallback: check if it's in main metadata (backward compatibility)
        main_metadata = self._load_main_metadata()
        if main_metadata.get("dataset_name") == dataset_name:
            logger.warning("Using legacy metadata format. Consider re-validating dataset.")
            return main_metadata
        
        raise FileNotFoundError(f"Metadata not found for dataset: {dataset_name}")
    
    def get_active_dataset(self) -> Optional[str]:
        """
        Get the currently active dataset name.
        
        Returns:
            Active dataset name or None if not set
        """
        try:
            main_metadata = self._load_main_metadata()
            return main_metadata.get("active_dataset")
        except FileNotFoundError:
            return None
    
    def set_active_dataset(self, dataset_name: str) -> None:
        """
        Set the active dataset.
        
        Args:
            dataset_name: Dataset identifier to set as active
            
        Raises:
            FileNotFoundError: If dataset metadata doesn't exist
        """
        # Verify dataset exists
        if not self._dataset_metadata_exists(dataset_name):
            raise FileNotFoundError(f"Dataset metadata not found: {dataset_name}")
        
        # Update main metadata
        main_metadata = self._load_main_metadata_safe()
        main_metadata["active_dataset"] = dataset_name
        
        self._save_main_metadata(main_metadata)
        logger.info(f"Active dataset set to: {dataset_name}")
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets with metadata.
        
        Returns:
            List of dataset metadata dictionaries
        """
        datasets = []
        
        # Ensure metadata directory exists
        if not self.metadata_dir.exists():
            return datasets
        
        # Read all dataset metadata files
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    dataset_meta = json.load(f)
                    datasets.append(dataset_meta)
            except Exception as e:
                logger.warning(f"Failed to load {metadata_file}: {e}")
        
        return sorted(datasets, key=lambda x: x.get("dataset_name", ""))
    
    def delete_dataset_metadata(self, dataset_name: str) -> None:
        """
        Delete metadata for a specific dataset.
        
        Args:
            dataset_name: Dataset identifier to delete
        """
        # Delete individual metadata file
        metadata_file = self._get_dataset_metadata_path(dataset_name)
        if metadata_file.exists():
            metadata_file.unlink()
            logger.info(f"Deleted metadata file: {metadata_file}")
        
        # Update main metadata
        main_metadata = self._load_main_metadata_safe()
        
        # If this was the active dataset, clear it
        if main_metadata.get("active_dataset") == dataset_name:
            main_metadata["active_dataset"] = None
            logger.info(f"Cleared active dataset (was: {dataset_name})")
        
        # Remove from datasets list if present
        if "datasets" in main_metadata and dataset_name in main_metadata["datasets"]:
            del main_metadata["datasets"][dataset_name]
        
        # Clear legacy fields if they match the deleted dataset
        if main_metadata.get("dataset_name") == dataset_name:
            main_metadata["dataset_name"] = None
            main_metadata["doc_count"] = None
            main_metadata["query_count"] = None
            main_metadata["qrels_count"] = None
        
        self._save_main_metadata(main_metadata)
    
    def metadata_exists(self, dataset_name: Optional[str] = None) -> bool:
        """
        Check if metadata exists for a dataset.
        
        Args:
            dataset_name: Dataset identifier. If None, checks for main metadata
            
        Returns:
            True if metadata file exists
        """
        if dataset_name is None:
            return self.path_resolver.get_metadata_path().exists()
        
        return self._dataset_metadata_exists(dataset_name)
    
    # ── Private Helper Methods ────────────────────────────────────────────
    
    def _get_dataset_metadata_path(self, dataset_name: str) -> Path:
        """Get path for individual dataset metadata file."""
        # Convert dataset name to safe filename
        safe_name = dataset_name.replace("/", "-").replace("\\", "-")
        return self.metadata_dir / f"{safe_name}.json"
    
    def _save_dataset_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        """Save individual dataset metadata file."""
        self.path_resolver.ensure_directory_exists(self.metadata_dir)
        
        metadata_file = self._get_dataset_metadata_path(dataset_name)
        self._write_json_with_retry(metadata_file, metadata)
    
    def _load_dataset_metadata(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Load individual dataset metadata file."""
        metadata_file = self._get_dataset_metadata_path(dataset_name)
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load dataset metadata {metadata_file}: {e}")
            return None
    
    def _dataset_metadata_exists(self, dataset_name: str) -> bool:
        """Check if individual dataset metadata file exists."""
        return self._get_dataset_metadata_path(dataset_name).exists()
    
    def _update_main_metadata(self, dataset_name: str, dataset_metadata: Dict[str, Any]) -> None:
        """Update main metadata file with dataset info and set as active."""
        main_metadata = self._load_main_metadata_safe()
        
        # Set as active dataset
        main_metadata["active_dataset"] = dataset_name
        
        # Initialize datasets dict if not present
        if "datasets" not in main_metadata:
            main_metadata["datasets"] = {}
        
        # Store summary info in main metadata
        main_metadata["datasets"][dataset_name] = {
            "dataset_name": dataset_name,
            "doc_count": dataset_metadata.get("doc_count"),
            "query_count": dataset_metadata.get("query_count"),
            "qrels_count": dataset_metadata.get("qrels_count"),
            "timestamp": dataset_metadata.get("timestamp"),
            "validation_status": dataset_metadata.get("validation_status"),
        }
        
        # Maintain backward compatibility fields from active dataset
        main_metadata["dataset_name"] = dataset_name
        main_metadata["doc_count"] = dataset_metadata.get("doc_count")
        main_metadata["query_count"] = dataset_metadata.get("query_count")
        main_metadata["qrels_count"] = dataset_metadata.get("qrels_count")
        main_metadata["timestamp"] = dataset_metadata.get("timestamp")
        main_metadata["validation_status"] = dataset_metadata.get("validation_status")
        main_metadata["ir_datasets_version"] = dataset_metadata.get("ir_datasets_version")
        
        self._save_main_metadata(main_metadata)
    
    def _load_main_metadata(self) -> Dict[str, Any]:
        """Load main metadata file."""
        metadata_path = self.path_resolver.get_metadata_path()
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Main metadata file not found: {metadata_path}")
        
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    
    def _load_main_metadata_safe(self) -> Dict[str, Any]:
        """Load main metadata file, return empty dict if not exists."""
        try:
            return self._load_main_metadata()
        except FileNotFoundError:
            return {}
    
    def _save_main_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save main metadata file."""
        metadata_path = self.path_resolver.get_metadata_path()
        self.path_resolver.ensure_directory_exists(metadata_path.parent)
        
        self._write_json_with_retry(metadata_path, metadata)
    
    def _write_json_with_retry(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON file with retry logic for Windows file locking."""
        import time
        
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return
                
            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Write attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to write {file_path} after {max_retries} attempts")
                    raise


# ── Convenience Functions ─────────────────────────────────────────────────

def get_metadata(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to get metadata for a dataset.
    
    Args:
        dataset_name: Dataset identifier. If None, returns active dataset metadata
        
    Returns:
        Dataset metadata dictionary
    """
    manager = DatasetMetadataManager()
    return manager.load_metadata(dataset_name)


def get_active_dataset_name() -> Optional[str]:
    """
    Get the name of the currently active dataset.
    
    Returns:
        Active dataset name or None
    """
    manager = DatasetMetadataManager()
    return manager.get_active_dataset()


def list_all_datasets() -> List[Dict[str, Any]]:
    """
    Get list of all datasets with metadata.
    
    Returns:
        List of dataset metadata dictionaries
    """
    manager = DatasetMetadataManager()
    return manager.list_datasets()
