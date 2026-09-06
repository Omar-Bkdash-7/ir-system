import json
from pathlib import Path
from common.config import get_config
import logging

logger = logging.getLogger(__name__)


def get_dataset_info() -> dict:
    """
    Get dataset metadata from the metadata directory.
    Uses the first available dataset metadata file as a fallback.
    
    Returns:
        Dictionary with dataset metadata
    """
    metadata_dir = Path(get_config("paths.metadata_dir", "data/metadata"))

    if not metadata_dir.exists():
        logger.warning(
            f"Metadata directory not found at {metadata_dir}. "
            "Returning config defaults. Run validation to create metadata."
        )
        return _get_fallback_metadata()

    try:
        metadata_files = list(metadata_dir.glob("*.json"))
        if not metadata_files:
            logger.warning("No metadata files found in metadata directory.")
            return _get_fallback_metadata()
        
        with open(metadata_files[0], encoding="utf-8") as f:
            metadata = json.load(f)
        logger.debug(f"Retrieved dataset info: {metadata.get('dataset_name')}")
        return metadata
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to read metadata: {e}. Falling back to config.")
        return _get_fallback_metadata()


def _get_fallback_metadata() -> dict:
    """
    Generate fallback metadata when no datasets are validated.
    
    Returns:
        Dictionary indicating no dataset is available
    """
    return {
        "dataset_name": None,
        "doc_count": 0,
        "query_count": 0,
        "qrels_count": 0,
        "validation_status": "not_validated",
        "source": "fallback",
        "note": "No datasets validated. Please validate a dataset first."
    }
