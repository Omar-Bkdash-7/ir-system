"""
Dataset Context Manager

Provides global access to current dataset information across all services.
This module maintains the concept of an "active dataset" that services
can query to determine which dataset they should operate on.

Usage:
    from common.dataset_context import get_active_dataset, get_dataset_metadata
    
    dataset_name = get_active_dataset()  # Returns 'beir/quora'
    metadata = get_dataset_metadata()    # Returns full metadata dict
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_active_dataset() -> Optional[str]:
    """
    Get the name of the currently active dataset.
    
    Returns:
        Active dataset name (e.g., 'beir/quora') or None if not set
        
    Example:
        >>> dataset = get_active_dataset()
        >>> print(dataset)
        'beir/quora'
    """
    try:
        from services.indexing.dataset.metadata_manager import get_active_dataset_name
        return get_active_dataset_name()
    except ImportError:
        logger.warning("Could not import metadata manager")
        return None
    except Exception as e:
        logger.warning(f"Failed to get active dataset: {e}")
        return None


def get_dataset_metadata(dataset_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific dataset or the active dataset.
    
    Args:
        dataset_name: Dataset identifier. If None, returns active dataset metadata
        
    Returns:
        Dataset metadata dictionary or None if not found
        
    Example:
        >>> metadata = get_dataset_metadata('beir/quora')
        >>> print(metadata['doc_count'])
        522931
    """
    try:
        from services.indexing.dataset.metadata_manager import get_metadata
        return get_metadata(dataset_name)
    except ImportError:
        logger.warning("Could not import metadata manager")
        return None
    except FileNotFoundError:
        logger.warning(f"Metadata not found for dataset: {dataset_name or 'active'}")
        return None
    except Exception as e:
        logger.warning(f"Failed to get dataset metadata: {e}")
        return None


def list_available_datasets() -> list:
    """
    Get list of all datasets with metadata.
    
    Returns:
        List of dataset metadata dictionaries
        
    Example:
        >>> datasets = list_available_datasets()
        >>> for ds in datasets:
        ...     print(ds['dataset_name'], ds['doc_count'])
        beir/quora 522931
        cranfield 1400
    """
    try:
        from services.indexing.dataset.metadata_manager import list_all_datasets
        return list_all_datasets()
    except ImportError:
        logger.warning("Could not import metadata manager")
        return []
    except Exception as e:
        logger.warning(f"Failed to list datasets: {e}")
        return []


def get_dataset_or_default(dataset_name: Optional[str] = None, default: str = "beir/quora") -> str:
    """
    Get dataset name with fallback chain.
    
    Priority:
    1. Provided dataset_name
    2. Active dataset from metadata
    3. Default value
    
    Args:
        dataset_name: Optional dataset identifier
        default: Default dataset to use as final fallback
        
    Returns:
        Dataset name string (never None)
        
    Example:
        >>> dataset = get_dataset_or_default()
        >>> # Returns active dataset, or 'beir/quora' if no active dataset
    """
    if dataset_name:
        return dataset_name
    
    active = get_active_dataset()
    if active:
        return active
    
    logger.info(f"No active dataset, using default: {default}")
    return default
