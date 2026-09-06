"""
Dataset Context Manager

Provides access to dataset information across services.
DEPRECATED: Use explicit dataset_name parameters instead of global context.

Usage (DEPRECATED):
    # Don't use this pattern anymore!
    # from common.dataset_context import get_active_dataset
    
    # Use this instead:
    def my_function(dataset_name: str):
        # Pass dataset_name explicitly
        pass
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def get_dataset_metadata(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific dataset.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        
    Returns:
        Dataset metadata dictionary or None if not found
        
    Example:
        >>> metadata = get_dataset_metadata('beir/quora')
        >>> print(metadata['doc_count'])
        522931
    """
    if not dataset_name:
        raise ValueError("dataset_name is required")
    
    try:
        from services.indexing.dataset.metadata_manager import get_metadata
        return get_metadata(dataset_name)
    except ImportError:
        logger.warning("Could not import metadata manager")
        return None
    except FileNotFoundError:
        logger.warning(f"Metadata not found for dataset: {dataset_name}")
        return None
    except Exception as e:
        logger.warning(f"Failed to get dataset metadata: {e}")
        return None


def list_available_datasets() -> List[Dict[str, Any]]:
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
