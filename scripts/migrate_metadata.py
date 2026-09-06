"""
Metadata Structure Checker

Checks the current metadata structure and lists all metadata files.

Usage:
    python scripts/migrate_metadata.py
"""

import json
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.indexing.dataset.metadata_manager import DatasetMetadataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_metadata_structure():
    """
    Check metadata structure and list current metadata files.
    """
    logger.info("=" * 80)
    logger.info("Checking Metadata Structure")
    logger.info("=" * 80)
    
    manager = DatasetMetadataManager()
    metadata_dir = manager.metadata_dir
    
    logger.info(f"\nMetadata directory: {metadata_dir}")
    
    # List current metadata files
    if metadata_dir.exists():
        metadata_files = list(metadata_dir.glob("*.json"))
        if metadata_files:
            logger.info(f"\nFound {len(metadata_files)} metadata file(s):")
            for mf in metadata_files:
                try:
                    with open(mf, encoding='utf-8') as f:
                        data = json.load(f)
                    dataset_name = data.get('dataset_name', 'unknown')
                    doc_count = data.get('doc_count', 0)
                    query_count = data.get('query_count', 0)
                    status = data.get('validation_status', 'unknown')
                    logger.info(f"  ✓ {mf.name}:")
                    logger.info(f"      Dataset: {dataset_name}")
                    logger.info(f"      Docs: {doc_count:,}, Queries: {query_count:,}")
                    logger.info(f"      Status: {status}")
                except Exception as e:
                    logger.warning(f"  ⚠ {mf.name}: Could not read ({e})")
        else:
            logger.info("\nNo metadata files found yet")
            logger.info("Run dataset validation to create metadata files")
    else:
        logger.info("\nMetadata directory does not exist yet")
        logger.info("It will be created when first dataset is validated")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Metadata structure check complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    check_metadata_structure()
