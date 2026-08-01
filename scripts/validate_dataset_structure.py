"""
Validate dataset storage structure.

This script checks if the dataset storage structure is correct
and provides recommendations for fixing any issues.
"""

import os
import json
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def validate_structure():
    """Validate the dataset storage structure."""
    
    logger.info("=" * 80)
    logger.info("Dataset Storage Structure Validation")
    logger.info("=" * 80)
    
    issues = []
    warnings = []
    
    data_dir = Path("data")
    datasets_dir = data_dir / "datasets"
    correct_location = datasets_dir / ".ir_datasets"
    wrong_location = datasets_dir / "ir_datasets"
    
    logger.info("\n1. Checking directory structure...")
    
    if not datasets_dir.exists():
        issues.append("❌ data/datasets/ directory does not exist")
    else:
        logger.info("   ✅ data/datasets/ exists")
    
    if not correct_location.exists():
        issues.append("❌ data/datasets/.ir_datasets/ does not exist")
    else:
        logger.info("   ✅ data/datasets/.ir_datasets/ exists (correct location)")
    
    if wrong_location.exists():
        warnings.append("⚠️  data/datasets/ir_datasets/ exists (duplicate location)")
        logger.warning("   ⚠️  Found duplicate location: data/datasets/ir_datasets/")
    else:
        logger.info("   ✅ No duplicate ir_datasets directory")
    
    logger.info("\n2. Checking for downloaded datasets...")
    
    if correct_location.exists():
        datasets = []
        for root, dirs, files in os.walk(correct_location):
            if "irds_version.txt" in files:
                root_path = Path(root)
                rel_path = root_path.relative_to(correct_location)
                dataset_id = str(rel_path).replace('\\', '/')
                
                size_mb = sum(
                    f.stat().st_size 
                    for f in root_path.rglob('*') 
                    if f.is_file()
                ) / 1024 / 1024
                
                datasets.append((dataset_id, size_mb))
        
        if datasets:
            logger.info(f"   ✅ Found {len(datasets)} dataset(s):")
            for ds_id, size in datasets:
                logger.info(f"      - {ds_id} ({size:.1f} MB)")
        else:
            warnings.append("⚠️  No datasets found in correct location")
            logger.warning("   ⚠️  No datasets downloaded yet")
    
    logger.info("\n3. Checking metadata.json...")
    
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            dataset_name = metadata.get("dataset_name", "unknown")
            status = metadata.get("validation_status", "unknown")
            
            logger.info(f"   ✅ metadata.json exists")
            logger.info(f"      Active dataset: {dataset_name}")
            logger.info(f"      Status: {status}")
            
            if status != "success":
                warnings.append(f"⚠️  Dataset validation status is '{status}'")
        except Exception as e:
            issues.append(f"❌ Failed to read metadata.json: {e}")
    else:
        warnings.append("⚠️  metadata.json does not exist (no dataset validated yet)")
    
    logger.info("\n4. Checking environment configuration...")
    
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            env_content = f.read()
        
        if "IR_DATASETS_HOME" in env_content:
            logger.info("   ✅ IR_DATASETS_HOME is set in .env")
        else:
            warnings.append("⚠️  IR_DATASETS_HOME not found in .env")
    else:
        warnings.append("⚠️  .env file does not exist")
    
    logger.info("\n5. Checking indices directory...")
    
    indices_dir = data_dir / "indices"
    if indices_dir.exists():
        index_dirs = [d for d in indices_dir.iterdir() if d.is_dir()]
        if index_dirs:
            logger.info(f"   ✅ Found {len(index_dirs)} index directories:")
            for idx_dir in index_dirs:
                logger.info(f"      - {idx_dir.name}")
        else:
            logger.info("   ℹ️  No indices built yet")
    else:
        warnings.append("⚠️  data/indices/ directory does not exist")
    
    logger.info("\n" + "=" * 80)
    logger.info("Validation Summary")
    logger.info("=" * 80)
    
    if not issues and not warnings:
        logger.info("\n✅ All checks passed! Dataset storage structure is correct.")
        return True
    
    if warnings:
        logger.info(f"\n⚠️  {len(warnings)} warning(s):")
        for warning in warnings:
            logger.warning(f"   {warning}")
    
    if issues:
        logger.error(f"\n❌ {len(issues)} issue(s) found:")
        for issue in issues:
            logger.error(f"   {issue}")
        
        logger.info("\n📋 Recommended actions:")
        logger.info("   1. Run: python scripts/fix_dataset_storage.py")
        logger.info("   2. Verify .env file has correct IR_DATASETS_HOME")
        logger.info("   3. Restart services if using Docker")
        
        return False
    
    logger.info("\n✅ Structure is correct, but some warnings were found.")
    logger.info("   Review warnings above and take action if needed.")
    return True


if __name__ == "__main__":
    try:
        success = validate_structure()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"\n❌ Validation failed with error: {e}", exc_info=True)
        exit(1)
