import json
from pathlib import Path
from common.config import get_config
import logging

logger = logging.getLogger(__name__)


def get_dataset_info() -> dict:
    """
    Get dataset metadata from file or fallback to config defaults.
    
    Returns:
        Dictionary with dataset metadata
    """
    metadata_path = Path(get_config("paths.metadata", "data/metadata.json"))

    if not metadata_path.exists():
        logger.warning(
            f"Dataset metadata not found at {metadata_path}. "
            "Returning config defaults. Run validation to create metadata."
        )
        return _get_fallback_metadata()

    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        logger.debug(f"Retrieved dataset info: {metadata.get('dataset_name')}")
        return metadata
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to read metadata: {e}. Falling back to config.")
        return _get_fallback_metadata()


def _get_fallback_metadata() -> dict:
    """
    Generate fallback metadata from configuration values.
    
    Returns:
        Dictionary with approximate dataset info from config
    """
    return {
        "dataset_name": get_config("dataset.name", "beir/quora"),
        "doc_count": get_config("dataset.expected_docs_count", 522931),
        "query_count": 10000,
        "qrels_count": 15675,
        "validation_status": "pending",
        "source": "config_fallback",
        "note": "Run dataset validation to get accurate counts"
    }
