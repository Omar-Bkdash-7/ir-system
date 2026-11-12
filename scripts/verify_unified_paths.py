#!/usr/bin/env python3
"""
Verify that all dataset paths are unified across the system.
"""

import os
import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_mark(passed: bool) -> str:
    """Return colored check mark or X based on test result."""
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 80}")
    print(f"{text}")
    print(f"{'=' * 80}{RESET}\n")

def verify_env_variable():
    """Verify IR_DATASETS_HOME environment variable."""
    print_header("1. Environment Variable Check")
    
    expected = "/app/data/datasets/.ir_datasets"
    actual = os.environ.get("IR_DATASETS_HOME")
    
    if actual == expected:
        print(f"{check_mark(True)} IR_DATASETS_HOME is set correctly")
        print(f"   Value: {GREEN}{actual}{RESET}")
        return True
    else:
        print(f"{check_mark(False)} IR_DATASETS_HOME mismatch!")
        print(f"   Expected: {GREEN}{expected}{RESET}")
        print(f"   Actual:   {RED}{actual or 'NOT SET'}{RESET}")
        return False

def verify_path_resolver():
    """Verify that path_resolver returns correct path."""
    print_header("2. Path Resolver Check")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from services.indexing.dataset.path_resolver import DatasetPathResolver
        
        resolver = DatasetPathResolver()
        ir_home = resolver.get_ir_home()
        
        expected_end = "data/datasets/.ir_datasets"
        actual_str = str(ir_home)
        
        if actual_str.endswith(expected_end):
            print(f"{check_mark(True)} Path resolver returns correct path")
            print(f"   Path: {GREEN}{ir_home}{RESET}")
            return True
        else:
            print(f"{check_mark(False)} Path resolver returns wrong path!")
            print(f"   Expected to end with: {GREEN}{expected_end}{RESET}")
            print(f"   Actual: {RED}{ir_home}{RESET}")
            return False
            
    except Exception as e:
        print(f"{check_mark(False)} Failed to import path_resolver: {e}")
        return False

def verify_directory_structure():
    """Verify that directory structure exists."""
    print_header("3. Directory Structure Check")
    
    base = Path("/app/data" if os.path.exists("/app/data") else "data")
    
    checks = [
        (base / "datasets", "datasets directory"),
        (base / "datasets" / ".ir_datasets", ".ir_datasets directory"),
        (base / "metadata", "metadata directory"),
    ]
    
    all_passed = True
    for path, description in checks:
        exists = path.exists()
        print(f"{check_mark(exists)} {description}")
        if exists:
            print(f"   {GREEN}{path}{RESET}")
        else:
            print(f"   {YELLOW}Not found: {path}{RESET}")
            all_passed = False
    
    return all_passed

def verify_docker_compose_config():
    """Verify docker-compose files have correct configuration."""
    print_header("4. Docker Compose Configuration Check")
    
    project_root = Path(__file__).parent.parent
    expected_value = "IR_DATASETS_HOME: /app/data/datasets/.ir_datasets"
    
    files_to_check = [
        "docker-compose.yml",
        "docker-compose.dev.yml",
    ]
    
    all_passed = True
    for filename in files_to_check:
        filepath = project_root / filename
        if not filepath.exists():
            print(f"{check_mark(False)} {filename} not found")
            all_passed = False
            continue
        
        content = filepath.read_text()
        if expected_value in content or "IR_DATASETS_HOME: /app/data/datasets/.ir_datasets" in content:
            print(f"{check_mark(True)} {filename} has correct IR_DATASETS_HOME")
        else:
            print(f"{check_mark(False)} {filename} missing or incorrect IR_DATASETS_HOME")
            all_passed = False
    
    return all_passed

def verify_env_file():
    """Verify .env file has correct configuration."""
    print_header("5. .env File Check")
    
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    if not env_file.exists():
        print(f"{check_mark(False)} .env file not found")
        return False
    
    content = env_file.read_text()
    expected = "IR_DATASETS_HOME=/app/data/datasets/.ir_datasets"
    
    if expected in content:
        print(f"{check_mark(True)} .env file has correct IR_DATASETS_HOME")
        return True
    else:
        print(f"{check_mark(False)} .env file has incorrect or missing IR_DATASETS_HOME")
        print(f"   Expected: {GREEN}{expected}{RESET}")
        return False

def verify_constants_file():
    """Verify DATA_FILE_EXTENSIONS includes .txt."""
    print_header("6. Constants Check")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from services.indexing.dataset.constants import DATA_FILE_EXTENSIONS
        
        if ".txt" in DATA_FILE_EXTENSIONS:
            print(f"{check_mark(True)} DATA_FILE_EXTENSIONS includes .txt")
            print(f"   Extensions: {GREEN}{', '.join(sorted(DATA_FILE_EXTENSIONS))}{RESET}")
            return True
        else:
            print(f"{check_mark(False)} DATA_FILE_EXTENSIONS missing .txt")
            print(f"   Extensions: {YELLOW}{', '.join(sorted(DATA_FILE_EXTENSIONS))}{RESET}")
            return False
            
    except Exception as e:
        print(f"{check_mark(False)} Failed to import constants: {e}")
        return False

def list_local_datasets():
    """List locally downloaded datasets."""
    print_header("7. Local Datasets")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from services.indexing.dataset.path_resolver import DatasetPathResolver
        from services.indexing.dataset.scanner import LocalDatasetScanner
        
        resolver = DatasetPathResolver()
        ir_home = resolver.get_ir_home()
        
        if not ir_home.exists():
            print(f"{YELLOW}⚠ IR_DATASETS_HOME does not exist yet{RESET}")
            print(f"   Path: {ir_home}")
            print("   This is normal if no datasets have been downloaded yet.")
            return True
        
        scanner = LocalDatasetScanner()
        datasets = scanner.list_datasets()
        
        if datasets:
            print(f"{GREEN}Found {len(datasets)} dataset(s):{RESET}")
            for ds in datasets:
                ds_id = ds.get('dataset_id', 'unknown')
                size_mb = ds.get('size_mb', 0)
                file_count = ds.get('file_count', 0)
                print(f"  • {ds_id}")
                print(f"    Size: {size_mb} MB, Files: {file_count}")
        else:
            print(f"{YELLOW}No datasets found (or empty directory){RESET}")
            print("   This is normal if no datasets have been downloaded yet.")
        
        return True
        
    except Exception as e:
        print(f"{check_mark(False)} Failed to scan datasets: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks."""
    print(f"\n{BLUE}{'=' * 80}")
    print("Dataset Paths Unified Verification")
    print(f"{'=' * 80}{RESET}\n")
    
    results = {
        "Environment Variable": verify_env_variable(),
        "Path Resolver": verify_path_resolver(),
        "Directory Structure": verify_directory_structure(),
        "Docker Compose Config": verify_docker_compose_config(),
        ".env File": verify_env_file(),
        "Constants File": verify_constants_file(),
        "Local Datasets": list_local_datasets(),
    }
    
    # Summary
    print_header("Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"Results: {passed}/{total} checks passed\n")
    
    for check_name, result in results.items():
        print(f"{check_mark(result)} {check_name}")
    
    if passed == total:
        print(f"\n{GREEN}✓ All checks passed! Dataset paths are unified.{RESET}\n")
        return 0
    else:
        print(f"\n{YELLOW}⚠ Some checks failed. Review the output above.{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
