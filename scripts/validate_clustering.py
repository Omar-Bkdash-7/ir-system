"""
Quick validation script for clustering service structure.
Checks that all imports work and the service can be initialized.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def validate_imports():
    """Check that all required modules can be imported."""
    print("Validating imports...")
    
    try:
        from services.clustering.schemas import (
            ClusterRequest,
            TopicRequest,
            ClusterResponse,
            ClusterInfo,
            TopicInfo,
            DocumentClusterInfo,
            ClusterListResponse,
        )
        print("✓ Schemas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import schemas: {e}")
        return False
    
    try:
        from services.clustering import clustering_service
        print("✓ Clustering service module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import clustering service: {e}")
        return False
    
    try:
        from services.clustering import main
        print("✓ Main API module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import main API: {e}")
        return False
    
    return True


def validate_structure():
    """Check that all required files exist."""
    print("\nValidating file structure...")
    
    required_files = [
        "services/clustering/__init__.py",
        "services/clustering/main.py",
        "services/clustering/clustering_service.py",
        "services/clustering/schemas/__init__.py",
        "services/clustering/schemas/requests.py",
        "services/clustering/schemas/responses.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist


def validate_config():
    """Check docker-compose and frontend config."""
    print("\nValidating configuration...")
    
    # Check docker-compose.yml
    docker_compose = project_root / "docker-compose.yml"
    if docker_compose.exists():
        content = docker_compose.read_text()
        if "clustering:" in content and "8007" in content:
            print("✓ Clustering service in docker-compose.yml")
        else:
            print("✗ Clustering service not properly configured in docker-compose.yml")
            return False
    else:
        print("✗ docker-compose.yml not found")
        return False
    
    # Check frontend config
    frontend_config = project_root / "frontend" / "config.py"
    if frontend_config.exists():
        content = frontend_config.read_text()
        if "CLUSTERING_URL" in content:
            print("✓ CLUSTERING_URL in frontend config")
        else:
            print("✗ CLUSTERING_URL not in frontend config")
            return False
    else:
        print("✗ frontend/config.py not found")
        return False
    
    # Check frontend page
    clustering_page = project_root / "frontend" / "app_pages" / "clustering.py"
    if clustering_page.exists():
        print("✓ Clustering page exists")
    else:
        print("✗ Clustering page not found")
        return False
    
    return True


def main():
    print("=" * 60)
    print("Clustering Service Validation")
    print("=" * 60)
    
    results = {
        "Structure": validate_structure(),
        "Imports": validate_imports(),
        "Configuration": validate_config(),
    }
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:20} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("\nThe clustering service is ready to be deployed!")
        print("\nNext steps:")
        print("1. Start the services: docker-compose -f docker-compose.yml -f docker-compose.dev.yml up")
        print("2. Build embeddings for a dataset via the Indexing page")
        print("3. Navigate to Clustering & Topics page")
        print("4. Run clustering and view results")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before deploying.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
