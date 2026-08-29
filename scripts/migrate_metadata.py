"""
Metadata Migration Script

Migrates from legacy single metadata.json to new multi-dataset structure.
Can be run safely multiple times (idempotent).

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


def migrate_legacy_metadata():
    """
    Migrate legacy metadata.json to new multi-dataset structure.
    """
    logger.info("=" * 80)
    logger.info("Starting metadata migration")
    logger.info("=" * 80)
    
    manager = DatasetMetadataManager()
    metadata_path = manager.path_resolver.get_metadata_path()
    
    # Check if legacy metadata exists
    if not metadata_path.exists():
        logger.info("No legacy metadata.json found. Nothing to migrate.")
        return
    
    # Load legacy metadata
    try:
        with open(metadata_path, encoding='utf-8') as f:
            legacy_metadata = json.load(f)
        logger.info(f"Loaded legacy metadata from: {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to load legacy metadata: {e}")
        return
    
    # Check if already migrated
    if "datasets" in legacy_metadata and "active_dataset" in legacy_metadata:
        logger.info("Metadata already migrated. Skipping.")
        return
    
    # Extract dataset name
    dataset_name = legacy_metadata.get("dataset_name")
    if not dataset_name:
        logger.warning("No dataset_name in legacy metadata. Cannot migrate.")
        return
    
    logger.info(f"Migrating dataset: {dataset_name}")
    
    # Save using new system (this creates individual file and updates main)
    try:
        manager.save_metadata(legacy_metadata, dataset_name=dataset_name)
        logger.info(f"✓ Successfully migrated metadata for: {dataset_name}")
        logger.info(f"  - Individual file: {manager._get_dataset_metadata_path(dataset_name)}")
        logger.info(f"  - Main file updated: {metadata_path}")
        logger.info(f"  - Active dataset set to: {dataset_name}")
    except Exception as e:
        logger.error(f"Failed to migrate metadata: {e}", exc_info=True)
        return
    
    # Verify migration
    try:
        active = manager.get_active_dataset()
        loaded_metadata = manager.load_metadata(dataset_name)
        
        logger.info("=" * 80)
        logger.info("Migration verification:")
        logger.info(f"  Active dataset: {active}")
        logger.info(f"  Metadata loaded for: {loaded_metadata.get('dataset_name')}")
        logger.info(f"  Doc count: {loaded_metadata.get('doc_count')}")
        logger.info("=" * 80)
        logger.info("✓ Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration verification failed: {e}")


if __name__ == "__main__":
    migrate_legacy_metadata()
