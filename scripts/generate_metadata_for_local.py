"""
Generate metadata for locally downloaded datasets that don't have metadata yet.

This script scans locally downloaded datasets and creates metadata for them
without re-downloading.
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.indexing.dataset.scanner import LocalDatasetScanner
from services.indexing.dataset.metadata_manager import DatasetMetadataManager
from services.indexing.dataset.loader import DatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_metadata_for_local_datasets():
    """Generate metadata for all local datasets that don't have metadata."""
    logger.info("=" * 80)
    logger.info("Generating metadata for local datasets")
    logger.info("=" * 80)
    
    # Get list of local datasets
    scanner = LocalDatasetScanner()
    local_datasets = scanner.list_datasets()
    
    logger.info(f"\nFound {len(local_datasets)} local dataset(s):")
    for ds in local_datasets:
        logger.info(f"  - {ds['dataset_id']}")
    
    # Get list of datasets with metadata
    metadata_manager = DatasetMetadataManager()
    datasets_with_meta = metadata_manager.list_datasets()
    datasets_with_meta_ids = {ds['dataset_name'] for ds in datasets_with_meta}
    
    logger.info(f"\nDatasets with metadata: {len(datasets_with_meta_ids)}")
    for ds_id in datasets_with_meta_ids:
        logger.info(f"  - {ds_id}")
    
    # Find datasets without metadata
    datasets_without_meta = [
        ds for ds in local_datasets 
        if ds['dataset_id'] not in datasets_with_meta_ids
    ]
    
    if not datasets_without_meta:
        logger.info("\n✓ All local datasets already have metadata!")
        return
    
    logger.info(f"\n⚠️  {len(datasets_without_meta)} dataset(s) need metadata:")
    for ds in datasets_without_meta:
        logger.info(f"  - {ds['dataset_id']}")
    
    # Generate metadata for each
    logger.info("\n" + "=" * 80)
    logger.info("Generating metadata...")
    logger.info("=" * 80)
    
    for ds_info in datasets_without_meta:
        dataset_id = ds_info['dataset_id']
        logger.info(f"\nProcessing: {dataset_id}")
        
        try:
            # Load dataset and get metadata
            loader = DatasetLoader(dataset_id)
            metadata = loader.load_with_retry(max_retries=3, retry_delay=5)
            
            # Save metadata (without setting as active)
            metadata_manager.save_metadata(metadata, dataset_name=dataset_id)
            
            logger.info(f"  ✓ Generated metadata for {dataset_id}")
            logger.info(f"    Docs: {metadata.get('doc_count', 0):,}")
            logger.info(f"    Queries: {metadata.get('query_count', 0):,}")
            logger.info(f"    Qrels: {metadata.get('qrels_count', 0):,}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to generate metadata for {dataset_id}: {e}")
            continue
    
    # Show summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    
    all_datasets = metadata_manager.list_datasets()
    active = metadata_manager.get_active_dataset()
    
    logger.info(f"Total datasets with metadata: {len(all_datasets)}")
    logger.info(f"Active dataset: {active}")
    
    for ds in all_datasets:
        is_active = " (ACTIVE)" if ds['dataset_name'] == active else ""
        logger.info(f"  - {ds['dataset_name']}{is_active}: {ds['doc_count']:,} docs")
    
    logger.info("\n✓ Done!")


if __name__ == "__main__":
    generate_metadata_for_local_datasets()
