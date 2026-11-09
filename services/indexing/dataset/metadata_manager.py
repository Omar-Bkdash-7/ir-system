"""
Dataset metadata management.

Supports multi-dataset metadata storage with individual files per dataset.
"""

import json
import logging
import os
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path

from common.config import get_config


logger = logging.getLogger(__name__)


class DatasetMetadataManager:
    """
    Manages multi-dataset metadata storage and retrieval.
    
    Features:
    - Each dataset has its own metadata file
    - No global state or active dataset concept
    - Atomic writes with retry logic for Windows compatibility
    
    Structure:
        data/
            metadata/               # Individual dataset metadata files
                beir-quora.json
                cranfield.json
                vaswani.json
    """
    
    def __init__(self):
        """Initialize metadata manager."""
        metadata_dir_path = get_config("paths.metadata_dir", "data/metadata")
        self.metadata_dir = Path(metadata_dir_path)
    
    def save_metadata(self, metadata: Dict[str, Any], dataset_name: str) -> None:
        """
        Save metadata for a specific dataset.
        
        Args:
            metadata: Dictionary containing dataset metadata
            dataset_name: Dataset identifier (e.g., 'beir/quora') - REQUIRED
            
        Raises:
            IOError: If file write fails
            ValueError: If dataset_name not provided
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        # Ensure metadata contains dataset_name
        metadata["dataset_name"] = dataset_name
        
        # Save individual dataset metadata
        self._ensure_dir_exists()
        self._save_dataset_metadata(dataset_name, metadata)
        
        logger.info(f"Metadata saved for dataset: {dataset_name}")
    
    def load_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load metadata for a specific dataset.
        
        Args:
            dataset_name: Dataset identifier (e.g., 'beir/quora') - REQUIRED
            
        Returns:
            Dictionary containing dataset metadata
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If metadata file is corrupted
            ValueError: If dataset_name not provided
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        # Load from individual file
        dataset_metadata = self._load_dataset_metadata(dataset_name)
        
        if dataset_metadata:
            return dataset_metadata
        
        raise FileNotFoundError(f"Metadata not found for dataset: {dataset_name}")
    
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
    
    def metadata_exists(self, dataset_name: str) -> bool:
        """
        Check if metadata exists for a dataset.
        
        Args:
            dataset_name: Dataset identifier
            
        Returns:
            True if metadata file exists
        """
        return self._dataset_metadata_exists(dataset_name)
    
    # ── Private Helper Methods ────────────────────────────────────────────
    
    def _ensure_dir_exists(self) -> None:
        """Ensure metadata directory exists."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_dataset_metadata_path(self, dataset_name: str) -> Path:
        """Get path for individual dataset metadata file."""
        # Convert dataset name to safe filename
        safe_name = dataset_name.replace("/", "-").replace("\\", "-")
        return self.metadata_dir / f"{safe_name}.json"
    
    def _save_dataset_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        """Save individual dataset metadata file."""
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
    
    def _write_json_with_retry(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON atomically (temp file + os.replace) with retry logic for Windows file locking.

        Writing the target file in place left a half-written JSON behind when the process died mid-write, and a
        reader (retrieval, gateway, the UI) loading at that moment got a decode error instead of the previous file.
        """
        import time
        
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            temp_path: Optional[Path] = None
            try:
                handle, temp_name = tempfile.mkstemp(prefix=f".{file_path.name}.", suffix=".tmp",
                                                     dir=str(file_path.parent))
                temp_path = Path(temp_name)
                with os.fdopen(handle, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(temp_path, file_path)  # atomic on POSIX and Windows (same directory)
                return
                
            except (OSError, IOError) as e:
                if temp_path is not None:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                if attempt < max_retries - 1:
                    logger.warning(f"Write attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to write {file_path} after {max_retries} attempts")
                    raise


# ── Convenience Functions ─────────────────────────────────────────────────

def get_metadata(dataset_name: str) -> Dict[str, Any]:
    """
    Convenience function to get metadata for a dataset.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        
    Returns:
        Dataset metadata dictionary
        
    Raises:
        ValueError: If dataset_name not provided
        FileNotFoundError: If metadata doesn't exist
    """
    if not dataset_name:
        raise ValueError("dataset_name is required")
    
    manager = DatasetMetadataManager()
    return manager.load_metadata(dataset_name)


def list_all_datasets() -> List[Dict[str, Any]]:
    """
    Get list of all datasets with metadata.
    
    Returns:
        List of dataset metadata dictionaries
    """
    manager = DatasetMetadataManager()
    return manager.list_datasets()


def known_dataset_ids() -> List[str]:
    """
    Exact ids of every dataset that has a metadata file (never raises).

    Used to turn PostgreSQL table names back into dataset ids: 'dataset_nano_beir_quora' alone cannot tell
    whether it was 'nano/beir/quora', 'nano-beir/quora' or 'nano_beir_quora'.
    """
    try:
        return [meta["dataset_name"] for meta in list_all_datasets() if meta.get("dataset_name")]
    except Exception as e:
        logger.warning(f"Could not list dataset metadata: {e}")
        return []
