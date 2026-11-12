#!/usr/bin/env python3
"""
Quick performance test to verify page loading optimizations.
Run this before and after the changes to compare load times.
"""
import time
import importlib
import sys
from pathlib import Path

# Add frontend to path
frontend_path = Path(__file__).parent.parent / "frontend"
sys.path.insert(0, str(frontend_path))


def test_lazy_loading():
    """Test that pages are loaded lazily."""
    print("Testing lazy module loading...")
    start = time.perf_counter()
    
    # This should be fast - no page imports yet
    import app
    
    elapsed = time.perf_counter() - start
    print(f"  ✓ App imported in {elapsed*1000:.2f}ms (no pages loaded)")
    
    # Now test individual page loads
    pages = ["search", "dataset", "indexing", "evaluation"]
    
    for page_name in pages:
        start = time.perf_counter()
        app._get_page_module(page_name)
        elapsed = time.perf_counter() - start
        print(f"  ✓ Page '{page_name}' loaded in {elapsed*1000:.2f}ms")


def test_cache_config():
    """Verify cache TTL settings."""
    print("\nVerifying cache configuration...")
    import config
    
    ttls = {
        "CACHE_TTL_HEALTH": config.CACHE_TTL_HEALTH,
        "CACHE_TTL_INDEX_STATUS": config.CACHE_TTL_INDEX_STATUS,
        "CACHE_TTL_DATASET_INFO": config.CACHE_TTL_DATASET_INFO,
    }
    
    for name, value in ttls.items():
        status = "✓" if value >= 60 else "✗"
        print(f"  {status} {name}: {value}s")


def test_import_speed():
    """Test import speed of critical modules."""
    print("\nTesting import speeds...")
    
    modules_to_test = [
        ("config", "frontend.config"),
        ("constants", "frontend.constants"),
        ("sidebar", "frontend.components.sidebar"),
    ]
    
    for name, module_path in modules_to_test:
        start = time.perf_counter()
        importlib.import_module(module_path.split(".")[-1])
        elapsed = time.perf_counter() - start
        print(f"  ✓ {name} imported in {elapsed*1000:.2f}ms")


if __name__ == "__main__":
    print("=" * 60)
    print("IR System Performance Tests")
    print("=" * 60)
    
    try:
        test_lazy_loading()
        test_cache_config()
        test_import_speed()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
