#!/usr/bin/env python3
"""
Verification script to check data paths configuration.
Run this inside a container to verify all paths are correctly set up.

Usage:
    docker exec ir_indexing python scripts/verify_data_paths.py
"""

import os
import sys
from pathlib import Path


def check_env_var(name: str, expected: str) -> bool:
    """Check if environment variable matches expected value."""
    actual = os.environ.get(name)
    
    if actual == expected:
        print(f"✅ {name}={actual}")
        return True
    else:
        print(f"❌ {name}={actual} (expected: {expected})")
        return False


def check_directory(path: str, should_exist: bool = True) -> bool:
    """Check if directory exists and is accessible."""
    p = Path(path)
    
    if should_exist:
        if p.exists() and p.is_dir():
            try:
                items = list(p.iterdir())
                print(f"✅ {path} exists ({len(items)} items)")
                return True
            except Exception as e:
                print(f"⚠️  {path} exists but not accessible: {e}")
                return False
        else:
            print(f"❌ {path} does not exist")
            return False
    else:
        if not p.exists():
            print(f"✅ {path} does not exist (as expected)")
            return True
        else:
            print(f"⚠️  {path} exists (but was not expected)")
            return True


def list_directory_contents(path: str, max_items: int = 10) -> None:
    """List directory contents for debugging."""
    p = Path(path)
    
    if not p.exists():
        print(f"   Directory does not exist: {path}")
        return
    
    if not p.is_dir():
        print(f"   Not a directory: {path}")
        return
    
    try:
        items = list(p.iterdir())
        print(f"   Contents of {path} ({len(items)} items):")
        
        for item in items[:max_items]:
            item_type = "DIR" if item.is_dir() else "FILE"
            size = ""
            if item.is_file():
                try:
                    size = f" ({item.stat().st_size:,} bytes)"
                except OSError:
                    pass
            print(f"     - {item.name} [{item_type}]{size}")
        
        if len(items) > max_items:
            print(f"     ... and {len(items) - max_items} more items")
            
    except Exception as e:
        print(f"   Error listing directory: {e}")


def main():
    print("=" * 80)
    print("Data Paths Verification Script")
    print("=" * 80)
    print()
    
    all_checks_passed = True
    
    # Check environment variables
    print("📋 Environment Variables:")
    print("-" * 80)
    
    checks = [
        ("IR_DATASETS_HOME", "/app/data/datasets/.ir_datasets"),
        ("SENTENCE_TRANSFORMERS_HOME", "/app/data/models"),
        ("TORCH_HOME", "/app/data/cache/torch"),
        ("TRANSFORMERS_CACHE", "/app/data/cache/transformers"),
    ]
    
    for name, expected in checks:
        if not check_env_var(name, expected):
            all_checks_passed = False
    
    print()
    
    # Check critical directories
    print("📁 Critical Directories:")
    print("-" * 80)
    
    dirs = [
        ("/app/data/datasets", True),
        ("/app/data/datasets/.ir_datasets", True),
        ("/app/data/indices", True),
        ("/app/data/cache", True),
        ("/app/data/processed", True),
        ("/app/data/evaluation", True),
        ("/app/data/models", True),
    ]
    
    for dir_path, should_exist in dirs:
        if not check_directory(dir_path, should_exist):
            all_checks_passed = False
    
    print()
    
    # Check metadata directory
    print("📄 Metadata Directory:")
    print("-" * 80)
    metadata_dir = "/app/data/metadata"
    if Path(metadata_dir).exists():
        print(f"✅ {metadata_dir} exists")
        try:
            metadata_files = list(Path(metadata_dir).glob("*.json"))
            if metadata_files:
                print(f"   Found {len(metadata_files)} metadata file(s):")
                import json
                for metadata_file in metadata_files[:5]:
                    with open(metadata_file) as f:
                        data = json.load(f)
                        print(f"   - {metadata_file.name}: {data.get('dataset_name', 'unknown')} ({data.get('doc_count', 0):,} docs)")
            else:
                print("   No metadata files found yet")
        except Exception as e:
            print(f"   ⚠️  Could not read metadata: {e}")
    else:
        print(f"⚠️  {metadata_dir} does not exist (will be created on first dataset validation)")
    
    print()
    
    # List contents of IR_DATASETS_HOME
    print("📦 IR_DATASETS_HOME Contents:")
    print("-" * 80)
    ir_home = os.environ.get("IR_DATASETS_HOME", "/app/data/datasets/.ir_datasets")
    list_directory_contents(ir_home, max_items=15)
    
    print()
    
    # Check for common dataset locations
    print("🔍 Checking for Common Datasets:")
    print("-" * 80)
    
    common_datasets = [
        "beir/quora",
        "msmarco-passage",
        "cranfield",
        "beir/scifact",
    ]
    
    for dataset_id in common_datasets:
        dataset_path = Path(ir_home) / dataset_id.replace('/', os.sep)
        if dataset_path.exists():
            print(f"✅ Found: {dataset_id}")
            list_directory_contents(str(dataset_path), max_items=5)
        else:
            print(f"⚠️  Not found: {dataset_id}")
    
    print()
    
    # Check indices directory structure
    print("📊 Indices Directory Structure:")
    print("-" * 80)
    indices_dir = Path("/app/data/indices")
    if indices_dir.exists():
        list_directory_contents(str(indices_dir), max_items=10)
        
        # Check if any indices exist
        try:
            index_dirs = [d for d in indices_dir.iterdir() if d.is_dir()]
            if index_dirs:
                print()
                print("   Checking index contents:")
                for idx_dir in index_dirs[:3]:  # Check first 3
                    print(f"   📂 {idx_dir.name}:")
                    list_directory_contents(str(idx_dir), max_items=5)
        except Exception as e:
            print(f"   Error checking indices: {e}")
    else:
        print("   Indices directory does not exist")
    
    print()
    
    # Summary
    print("=" * 80)
    if all_checks_passed:
        print("✅ All critical checks passed!")
        print("   Your data paths are correctly configured.")
        return 0
    else:
        print("❌ Some checks failed!")
        print("   Please review the output above and fix the configuration.")
        print("   See DATA_PATHS_DOCUMENTATION.md for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
