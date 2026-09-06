"""
Utility functions for dataset operations in the frontend.
"""
from pathlib import Path
from typing import List
import os
import logging

logger = logging.getLogger(__name__)


def get_available_datasets() -> List[str]:
    """
    Get list of datasets that are:
    1. Downloaded locally (exist in data/datasets/.ir_datasets)
    2. Have all three indices built (tfidf, bm25, embedding)
    
    Returns:
        List of dataset names (e.g., ['cranfield', 'vaswani'])
    """
    # Try multiple possible root paths (for Docker and local development)
    possible_roots = [
        Path(__file__).parent.parent.parent.resolve(),  # From frontend/utils
        Path.cwd(),  # Current working directory
        Path("/app") if Path("/app").exists() else None,  # Docker environment
    ]
    
    datasets_dir = None
    indices_dir = None
    
    for root in possible_roots:
        if root is None:
            continue
        test_datasets_dir = root / "data" / "datasets" / ".ir_datasets"
        test_indices_dir = root / "data" / "indices"
        
        if test_datasets_dir.exists() and test_indices_dir.exists():
            datasets_dir = test_datasets_dir
            indices_dir = test_indices_dir
            break
    
    available_datasets = []
    
    if not datasets_dir or not indices_dir:
        logger.warning("Could not find datasets or indices directories")
        return available_datasets
    
    try:
        for dataset_path in datasets_dir.iterdir():
            if not dataset_path.is_dir():
                continue
                
            dataset_name = dataset_path.name
            
            if not _has_dataset_content(dataset_path):
                continue
            
            if _has_all_indices(indices_dir, dataset_name):
                available_datasets.append(dataset_name)
    except (PermissionError, OSError) as e:
        logger.error(f"Error scanning datasets: {e}")
    
    return sorted(available_datasets)


def _has_dataset_content(dataset_path: Path) -> bool:
    """
    Check if dataset directory has actual content (not empty).
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        True if dataset has content, False otherwise
    """
    if not dataset_path.exists() or not dataset_path.is_dir():
        return False
    
    try:
        content = list(dataset_path.iterdir())
        return len(content) > 0
    except (PermissionError, OSError):
        return False


def _has_all_indices(indices_dir: Path, dataset_name: str) -> bool:
    """
    Check if all three indices (tfidf, bm25, embedding) exist for a dataset.
    
    Args:
        indices_dir: Path to indices directory
        dataset_name: Name of the dataset
        
    Returns:
        True if all indices exist, False otherwise
    """
    dataset_indices_dir = indices_dir / dataset_name
    
    if not dataset_indices_dir.exists():
        return False
    
    required_indices = {"tfidf", "bm25", "embedding"}
    
    try:
        existing_indices = {
            item.name.lower()
            for item in dataset_indices_dir.iterdir() 
            if item.is_dir()
        }
        
        has_all = required_indices.issubset(existing_indices)
        return has_all
    except (PermissionError, OSError) as e:
        logger.error(f"Error checking indices for {dataset_name}: {e}")
        return False


def get_dataset_display_name(dataset_name: str) -> str:
    """
    Convert dataset name to a more readable display format.
    
    Args:
        dataset_name: Internal dataset name (e.g., 'cranfield', 'beir_arguana')
        
    Returns:
        Display name (e.g., 'Cranfield', 'BEIR: Arguana')
    """
    if dataset_name.startswith("beir"):
        parts = dataset_name.split("_")
        if len(parts) > 1:
            sub_dataset = " ".join(parts[1:]).title()
            return f"BEIR: {sub_dataset}"
        return "BEIR"
    
    if dataset_name.startswith("trec"):
        parts = dataset_name.split("_", 1)
        if len(parts) > 1:
            sub_dataset = parts[1].replace("_", " ").title()
            return f"TREC: {sub_dataset}"
        return "TREC"
    
    return dataset_name.replace("_", " ").title()
