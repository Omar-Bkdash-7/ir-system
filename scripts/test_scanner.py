"""
Test dataset scanner functionality
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.indexing.dataset.scanner import LocalDatasetScanner
from services.indexing.dataset.path_resolver import DatasetPathResolver

def main():
    print("=" * 70)
    print("Testing Dataset Scanner")
    print("=" * 70)
    
    # Test path resolver
    resolver = DatasetPathResolver()
    ir_home = resolver.get_ir_home()
    print(f"\nIR_DATASETS_HOME: {ir_home}")
    print(f"Exists: {ir_home.exists()}")
    
    if ir_home.exists():
        print(f"\nContents of {ir_home}:")
        try:
            for item in sorted(ir_home.iterdir()):
                print(f"  - {item.name} {'[DIR]' if item.is_dir() else '[FILE]'}")
        except Exception as e:
            print(f"  Error listing: {e}")
    
    # Test scanner
    print("\n" + "=" * 70)
    print("Running Dataset Scanner")
    print("=" * 70)
    
    scanner = LocalDatasetScanner()
    datasets = scanner.list_datasets()
    
    print(f"\nFound {len(datasets)} dataset(s):\n")
    
    for ds in datasets:
        print(f"  Dataset ID: {ds['dataset_id']}")
        print(f"  Path: {ds['path']}")
        print(f"  Size: {ds['size_mb']} MB")
        print(f"  Files: {ds['file_count']}")
        print()
    
    if not datasets:
        print("  [No datasets found]")
        print("\n  Troubleshooting:")
        print("  1. Check if datasets are downloaded")
        print("  2. Verify IR_DATASETS_HOME path")
        print("  3. Check file permissions")

if __name__ == "__main__":
    main()
