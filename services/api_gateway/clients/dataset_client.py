import json
from pathlib import Path
from common.config import get_config
import logging

logger = logging.getLogger(__name__)


def get_dataset_info() -> dict:
    metadata_path = Path(get_config("paths.metadata", "data/metadata.json"))

    if not metadata_path.exists():
        logger.error(f"Dataset metadata not found at {metadata_path}")
        raise FileNotFoundError("Dataset metadata not found")

    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.debug(f"Retrieved dataset info: {metadata.get('dataset_name')}")
    return metadata
