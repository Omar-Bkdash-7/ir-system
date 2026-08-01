"""
Script to fix and reorganize dataset storage structure.

This script:
1. Identifies duplicate dataset storage locations
2. Consolidates datasets to the correct location (.ir_datasets)
3. Removes redundant duplicate directories
4. Updates metadata.json with correct information
"""

import os
import shutil
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_datasets_in_directory(base_path: Path) -> dict:
    """
    Scan directory for valid datasets.
    
    Args:
        base_path: Base directory to scan
        
    Returns:
        Dict mapping dataset_id to dataset info
    """
    datasets = {}
    
    if not base_path.exists():
        logger.warning(f"Directory does not exist: {base_path}")
        return datasets
    
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        
        has_version = (root_path / "irds_version.txt").exists()
        has_queries = any(
            (root_path / f).exists() 
            for f in ["queries.json", "queries.jsonl", "queries.tsv"]
        )
        has_qrels = any(
            f.endswith('.qrels') or f == 'qrels' 
            for f in files
        )
        has_docs = any(
            d.startswith('docs') for d in dirs
        ) or any(
            f.endswith(('.pklz4', '.jsonl', '.tsv', '.zip')) and 'source' not in f 
            for f in files
        )
        
        if has_version and (has_queries or has_docs or has_qrels):
            rel_path = root_path.relative_to(base_path)
            dataset_id = str(rel_path).replace('\\', '/')
            
            if dataset_id and dataset_id != '.':
                total_size = sum(
                    f.stat().st_size 
                    for f in root_path.rglob('*') 
                    if f.is_file()
                )
                
                datasets[dataset_id] = {
                    'path': root_path,
                    'size_bytes': total_size,
                    'has_version': has_version,
                    'has_queries': has_queries,
                    'has_qrels': has_qrels,
                    'has_docs': has_docs
                }
                
                logger.info(
                    f"Found dataset '{dataset_id}' at {root_path} "
                    f"({total_size / 1024 / 1024:.1f} MB)"
                )
    
    return datasets


def consolidate_datasets():
    """Main function to consolidate datasets to correct location."""
    
    data_dir = Path("data")
    datasets_dir = data_dir / "datasets"
    
    correct_location = datasets_dir / ".ir_datasets"
    duplicate_location = datasets_dir / "ir_datasets"
    
    logger.info("=" * 80)
    logger.info("Dataset Storage Consolidation Script")
    logger.info("=" * 80)
    
    logger.info(f"\nScanning correct location: {correct_location}")
    correct_datasets = find_datasets_in_directory(correct_location)
    
    logger.info(f"\nScanning duplicate location: {duplicate_location}")
    duplicate_datasets = find_datasets_in_directory(duplicate_location)
    
    if not duplicate_datasets:
        logger.info("\n✅ No duplicate datasets found. Storage structure is correct.")
        logger.info(f"Found {len(correct_datasets)} dataset(s) in correct location.")
        return
    
    logger.info(f"\n⚠️  Found {len(duplicate_datasets)} dataset(s) in duplicate location")
    logger.info("=" * 80)
    
    for dataset_id, info in duplicate_datasets.items():
        logger.info(f"\nProcessing: {dataset_id}")
        
        duplicate_path = info['path']
        correct_path = correct_location / dataset_id
        
        if dataset_id in correct_datasets:
            correct_size = correct_datasets[dataset_id]['size_bytes']
            duplicate_size = info['size_bytes']
            
            logger.info(f"  Dataset exists in both locations:")
            logger.info(f"    Correct:   {correct_size / 1024 / 1024:.1f} MB")
            logger.info(f"    Duplicate: {duplicate_size / 1024 / 1024:.1f} MB")
            
            if correct_size >= duplicate_size:
                logger.info(f"  ✅ Correct location has complete data. Removing duplicate...")
                try:
                    shutil.rmtree(duplicate_path)
                    logger.info(f"  ✅ Removed duplicate at {duplicate_path}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to remove duplicate: {e}")
            else:
                logger.warning(f"  ⚠️  Duplicate has more data. Moving to correct location...")
                try:
                    if correct_path.exists():
                        shutil.rmtree(correct_path)
                    
                    correct_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(duplicate_path), str(correct_path))
                    logger.info(f"  ✅ Moved to {correct_path}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to move dataset: {e}")
        else:
            logger.info(f"  Dataset only exists in duplicate location. Moving...")
            try:
                correct_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(duplicate_path), str(correct_path))
                logger.info(f"  ✅ Moved to {correct_path}")
            except Exception as e:
                logger.error(f"  ❌ Failed to move dataset: {e}")
    
    if duplicate_location.exists() and not any(duplicate_location.iterdir()):
        logger.info(f"\n🗑️  Removing empty duplicate directory: {duplicate_location}")
        try:
            duplicate_location.rmdir()
            logger.info("  ✅ Removed empty directory")
        except Exception as e:
            logger.error(f"  ❌ Failed to remove directory: {e}")
    
    cleanup_empty_parents(duplicate_location.parent, datasets_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("Consolidation complete!")
    
    final_datasets = find_datasets_in_directory(correct_location)
    logger.info(f"✅ Total datasets in correct location: {len(final_datasets)}")
    
    for dataset_id, info in final_datasets.items():
        size_mb = info['size_bytes'] / 1024 / 1024
        logger.info(f"  - {dataset_id}: {size_mb:.1f} MB")
    
    logger.info("=" * 80)


def cleanup_empty_parents(path: Path, stop_at: Path):
    """Remove empty parent directories up to stop_at."""
    current = path
    while current != stop_at and current.exists():
        try:
            if not any(current.iterdir()):
                logger.info(f"  Removing empty directory: {current}")
                current.rmdir()
                current = current.parent
            else:
                break
        except Exception as e:
            logger.debug(f"  Could not remove {current}: {e}")
            break


def update_environment_variable():
    """Display instructions for setting IR_DATASETS_HOME."""
    correct_path = Path("data/datasets/.ir_datasets").absolute()
    
    logger.info("\n" + "=" * 80)
    logger.info("Environment Variable Configuration")
    logger.info("=" * 80)
    logger.info("\nTo ensure ir_datasets uses the correct location, set:")
    logger.info(f"\n  IR_DATASETS_HOME={correct_path}")
    logger.info("\nFor Docker, this is already set in docker-compose.yml")
    logger.info("For local development, add to your .env file:")
    logger.info(f"\n  IR_DATASETS_HOME={correct_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        consolidate_datasets()
        update_environment_variable()
        
        logger.info("\n✅ Dataset storage structure has been fixed!")
        logger.info("All datasets are now in: data/datasets/.ir_datasets/")
        
    except Exception as e:
        logger.error(f"\n❌ Script failed with error: {e}", exc_info=True)
        exit(1)
