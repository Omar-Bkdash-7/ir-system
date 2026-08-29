"""
Test Multi-Dataset System

Test script to verify multi-dataset metadata management.

Usage:
    docker exec ir_indexing python scripts/test_multi_dataset.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.indexing.dataset.metadata_manager import DatasetMetadataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_multi_dataset_system():
    """Test multi-dataset metadata system."""
    logger.info("=" * 80)
    logger.info("Testing Multi-Dataset System")
    logger.info("=" * 80)
    
    manager = DatasetMetadataManager()
    
    # Test 1: Get active dataset
    logger.info("\n[Test 1] Get Active Dataset")
    try:
        active = manager.get_active_dataset()
        logger.info(f"  ✓ Active dataset: {active}")
    except Exception as e:
        logger.warning(f"  ⚠ No active dataset: {e}")
    
    # Test 2: List all datasets
    logger.info("\n[Test 2] List All Datasets")
    try:
        datasets = manager.list_datasets()
        logger.info(f"  ✓ Found {len(datasets)} dataset(s)")
        for ds in datasets:
            ds_name = ds.get("dataset_name")
            doc_count = ds.get("doc_count", 0)
            logger.info(f"    - {ds_name}: {doc_count:,} documents")
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
    
    # Test 3: Load specific dataset metadata
    logger.info("\n[Test 3] Load Specific Dataset")
    if datasets:
        test_dataset = datasets[0].get("dataset_name")
        try:
            metadata = manager.load_metadata(test_dataset)
            logger.info(f"  ✓ Loaded metadata for: {test_dataset}")
            logger.info(f"    Documents: {metadata.get('doc_count', 0):,}")
            logger.info(f"    Queries: {metadata.get('query_count', 0):,}")
            logger.info(f"    Status: {metadata.get('validation_status')}")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {test_dataset}: {e}")
    
    # Test 4: Check metadata file structure
    logger.info("\n[Test 4] Check File Structure")
    metadata_dir = manager.metadata_dir
    main_metadata = manager.path_resolver.get_metadata_path()
    
    logger.info(f"  Main metadata file: {main_metadata}")
    logger.info(f"    Exists: {main_metadata.exists()}")
    
    logger.info(f"  Metadata directory: {metadata_dir}")
    logger.info(f"    Exists: {metadata_dir.exists()}")
    
    if metadata_dir.exists():
        metadata_files = list(metadata_dir.glob("*.json"))
        logger.info(f"    Files: {len(metadata_files)}")
        for f in metadata_files:
            logger.info(f"      - {f.name}")
    
    # Test 5: Test convenience functions
    logger.info("\n[Test 5] Test Convenience Functions")
    try:
        from services.indexing.dataset.metadata_manager import (
            get_metadata,
            get_active_dataset_name,
            list_all_datasets
        )
        
        active_name = get_active_dataset_name()
        logger.info(f"  ✓ get_active_dataset_name(): {active_name}")
        
        if active_name:
            active_meta = get_metadata()
            logger.info(f"  ✓ get_metadata(): {active_meta.get('dataset_name')}")
        
        all_ds = list_all_datasets()
        logger.info(f"  ✓ list_all_datasets(): {len(all_ds)} datasets")
        
    except Exception as e:
        logger.error(f"  ✗ Convenience functions failed: {e}")
    
    # Test 6: Test dataset context helper
    logger.info("\n[Test 6] Test Dataset Context Helper")
    try:
        from common.dataset_context import (
            get_active_dataset,
            get_dataset_metadata,
            get_dataset_or_default
        )
        
        ctx_active = get_active_dataset()
        logger.info(f"  ✓ context.get_active_dataset(): {ctx_active}")
        
        if ctx_active:
            ctx_meta = get_dataset_metadata(ctx_active)
            logger.info(f"  ✓ context.get_dataset_metadata(): docs={ctx_meta.get('doc_count', 0):,}")
        
        default_ds = get_dataset_or_default()
        logger.info(f"  ✓ context.get_dataset_or_default(): {default_ds}")
        
    except Exception as e:
        logger.error(f"  ✗ Dataset context failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Testing Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_multi_dataset_system()
