"""
Fix corrupted dataset by re-downloading it completely.

This script:
1. Detects corrupted dataset files
2. Removes them completely
3. Re-downloads the dataset from ir_datasets registry
"""
import logging
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ir_datasets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_dataset_path(dataset_id: str) -> Path:
    """Get path to dataset directory."""
    ir_home = os.environ.get("IR_DATASETS_HOME", "data/datasets/.ir_datasets")
    dataset_path = Path(ir_home)
    
    # Build path from dataset ID
    parts = dataset_id.strip("/").split("/")
    for part in parts:
        dataset_path = dataset_path / part
    
    return dataset_path


def check_dataset_corruption(dataset_id: str) -> bool:
    """
    Check if dataset is corrupted by trying to iterate documents.
    
    Returns:
        True if corrupted, False if okay
    """
    logger.info(f"Checking dataset '{dataset_id}' for corruption...")
    
    try:
        ds = ir_datasets.load(dataset_id)
        
        if not hasattr(ds, 'docs_iter'):
            logger.warning(f"Dataset '{dataset_id}' has no docs_iter")
            return False
        
        # Try to read first 10 documents
        for i, doc in enumerate(ds.docs_iter()):
            if i >= 10:
                break
        
        logger.info(f"✓ Dataset '{dataset_id}' appears to be intact")
        return False
        
    except Exception as e:
        logger.error(f"✗ Dataset '{dataset_id}' is corrupted: {e}")
        return True


def fix_corrupted_dataset(dataset_id: str, force: bool = False) -> bool:
    """
    Fix corrupted dataset by deleting and re-downloading.
    
    Args:
        dataset_id: Dataset identifier (e.g., 'beir/arguana')
        force: Force re-download even if not corrupted
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info(f"Fixing dataset: {dataset_id}")
    logger.info("=" * 80)
    
    # Check if corrupted (unless forced)
    if not force:
        is_corrupted = check_dataset_corruption(dataset_id)
        if not is_corrupted:
            logger.info(f"Dataset '{dataset_id}' is not corrupted. Use --force to re-download anyway.")
            return True
    
    # Get dataset path
    dataset_path = get_dataset_path(dataset_id)
    
    logger.info(f"Dataset path: {dataset_path}")
    
    # Delete existing dataset
    if dataset_path.exists():
        logger.warning(f"Deleting corrupted dataset at: {dataset_path}")
        try:
            shutil.rmtree(dataset_path)
            logger.info("✓ Deleted corrupted dataset files")
        except Exception as e:
            logger.error(f"✗ Failed to delete dataset: {e}")
            return False
    else:
        logger.info("Dataset directory does not exist locally")
    
    # Re-download dataset
    logger.info(f"Re-downloading dataset '{dataset_id}'...")
    
    try:
        # Load dataset (this will trigger download)
        ds = ir_datasets.load(dataset_id)
        
        # Force complete download by iterating
        logger.info("Iterating documents to force complete download...")
        doc_count = 0
        
        if hasattr(ds, 'docs_iter'):
            for doc in ds.docs_iter():
                doc_count += 1
                if doc_count % 1000 == 0:
                    logger.info(f"  Downloaded {doc_count:,} documents...")
        
        logger.info(f"✓ Successfully downloaded {doc_count:,} documents")
        
        # Verify queries and qrels
        if hasattr(ds, 'queries_iter'):
            query_count = sum(1 for _ in ds.queries_iter())
            logger.info(f"✓ Verified {query_count:,} queries")
        
        if hasattr(ds, 'qrels_iter'):
            qrel_count = sum(1 for _ in ds.qrels_iter())
            logger.info(f"✓ Verified {qrel_count:,} qrels")
        
        logger.info("=" * 80)
        logger.info(f"✓ Dataset '{dataset_id}' fixed successfully!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to re-download dataset: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix corrupted ir_datasets dataset")
    parser.add_argument("dataset_id", help="Dataset ID (e.g., 'beir/arguana')")
    parser.add_argument("--force", action="store_true", help="Force re-download even if not corrupted")
    
    args = parser.parse_args()
    
    success = fix_corrupted_dataset(args.dataset_id, force=args.force)
    
    if success:
        logger.info("\n✓ Done! You can now upload the dataset to database.")
        sys.exit(0)
    else:
        logger.error("\n✗ Failed to fix dataset. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
