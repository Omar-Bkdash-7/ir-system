"""
Quick validation script for the clustering service structure.
Checks that the modules import, the package layout is complete, and the service is wired into configuration.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

REQUIRED_FILES = [
    "services/clustering/__init__.py",
    "services/clustering/main.py",
    "services/clustering/config.py",
    "services/clustering/errors.py",
    "services/clustering/service.py",
    "services/clustering/core/__init__.py",
    "services/clustering/core/paths.py",
    "services/clustering/core/embeddings.py",
    "services/clustering/core/texts.py",
    "services/clustering/core/clusterer.py",
    "services/clustering/core/topics.py",
    "services/clustering/core/store.py",
    "services/clustering/schemas/__init__.py",
    "services/clustering/schemas/requests.py",
    "services/clustering/schemas/responses.py",
    "tests/clustering/test_service.py",
    "tests/clustering/test_http.py",
]


def validate_structure():
    print("\nValidating file structure...")
    missing = [path for path in REQUIRED_FILES if not (project_root / path).exists()]
    for path in REQUIRED_FILES:
        print(("✓ " if (project_root / path).exists() else "✗ ") + path + (" NOT FOUND" if path in missing else ""))
    return not missing


def validate_imports():
    print("\nValidating imports...")
    try:
        from services.clustering.schemas import (  # noqa: F401
            ClusterRequest,
            ClusterResponse,
            ClusterInfo,
            ClusterParams,
            ClusterStatusResponse,
            TopicInfo,
            DocumentClusterInfo,
            ClusterListResponse,
        )
        from services.clustering.service import ClusteringService  # noqa: F401
        from services.clustering import main  # noqa: F401
        from services.clustering.core import clusterer  # noqa: F401
    except ImportError as exc:
        print(f"✗ Import failed: {exc}")
        return False
    print("✓ Schemas, service and app import")
    print(f"✓ HDBSCAN/UMAP available in this environment: {clusterer.CLUSTERING_AVAILABLE}")
    return True


def validate_config():
    print("\nValidating configuration...")
    checks = []

    compose = project_root / "docker-compose.yml"
    content = compose.read_text(encoding="utf-8") if compose.exists() else ""
    checks.append(("clustering service in docker-compose.yml", "  clustering:" in content and "8007" in content))

    config_yaml = project_root / "configs" / "config.yaml"
    yaml_content = config_yaml.read_text(encoding="utf-8") if config_yaml.exists() else ""
    checks.append(("clustering: section in configs/config.yaml", "\nclustering:" in yaml_content))

    frontend_config = project_root / "frontend" / "config.py"
    frontend_content = frontend_config.read_text(encoding="utf-8") if frontend_config.exists() else ""
    checks.append(("CLUSTERING_URL in frontend/config.py", "CLUSTERING_URL" in frontend_content))

    page = project_root / "frontend" / "app_pages" / "clustering.py"
    checks.append(("clustering page exists", page.exists()))

    for label, passed in checks:
        print(("✓ " if passed else "✗ ") + label)
    return all(passed for _, passed in checks)


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
        print(f"{check:20} {'✓ PASS' if passed else '✗ FAIL'}")

    all_passed = all(results.values())
    print("\n" + ("✓ ALL CHECKS PASSED" if all_passed else "✗ SOME CHECKS FAILED"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
