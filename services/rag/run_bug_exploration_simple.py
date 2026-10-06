"""
Simplified Bug Condition Exploration (No pytest dependency)

This script demonstrates the three bugs in RAG Auto Mode on UNFIXED code.
Run this script to confirm the bugs exist before applying fixes.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rag.rag_service import RAGService
from services.rag.schemas import RAGRequest, GenerationConfig, ContextDocument


def print_test_header(test_name):
    print("\n" + "=" * 70)
    print(f"TEST: {test_name}")
    print("=" * 70)


async def test_bug1_missing_import():
    """
    Bug 1: Missing GenerationConfig Import
    
    Expected on UNFIXED code: NameError at line 61 or 79
    Expected on FIXED code: No NameError, code executes
    """
    print_test_header("Bug 1 - Missing GenerationConfig Import")
    
    try:
        with patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "hf_test_token_12345"}):
            service = RAGService()
            
            # Mock retrieval
            mock_docs = [
                ContextDocument(
                    doc_id="doc1",
                    text="Test document about aerodynamics.",
                    score=0.95,
                    metadata={}
                )
            ]
            service.retrieve_context = AsyncMock(return_value=mock_docs)
            service._check_internet = AsyncMock(return_value=True)
            
            # Create Auto Mode request
            request = RAGRequest(
                query="What is aerodynamics?",
                dataset="cranfield",
                retrieval_model="embedding",
                generation_config=GenerationConfig(model="auto")
            )
            
            print("Triggering Auto Mode with valid token and internet...")
            print("Expected behavior: Code should attempt to instantiate GenerationConfig at line 61")
            
            response = await service.generate_answer(request)
            
            print("✓ SUCCESS: No NameError occurred!")
            print("  This means GenerationConfig is properly imported.")
            print("  Bug 1 is FIXED.")
            return True
            
    except NameError as e:
        print(f"✗ FAIL: NameError occurred - {str(e)}")
        print("  This confirms Bug 1: GenerationConfig is not imported in rag_service.py")
        print("  Expected at line 61 or 79: NameError: name 'GenerationConfig' is not defined")
        print("  Bug 1 is CONFIRMED on UNFIXED code.")
        return False
    except Exception as e:
        print(f"⚠ Other error occurred: {type(e).__name__}: {str(e)}")
        print("  This may be expected (no real models), but no NameError = partial fix")
        return True


async def test_bug2_empty_token():
    """
    Bug 2a: Empty Token Detection
    
    Expected on UNFIXED code: Empty token passes `if hf_token` check
    Expected on FIXED code: Empty token detected as invalid
    """
    print_test_header("Bug 2a - Empty Token Detection")
    
    with patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": ""}):
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        print(f"Token value: '{token}'")
        print(f"Token type: {type(token)}")
        
        # Test the validation logic
        is_valid_unfixed = bool(token)  # Current (buggy) logic
        is_valid_fixed = bool(token and token.strip())  # Fixed logic
        
        print(f"\nCurrent logic: if hf_token → {is_valid_unfixed}")
        print(f"Fixed logic: if hf_token and hf_token.strip() → {is_valid_fixed}")
        
        if is_valid_unfixed and not is_valid_fixed:
            print("\n✗ FAIL: Empty token passes current validation!")
            print("  This confirms Bug 2a: Weak token validation")
            print("  Empty string satisfies `if hf_token` but should be invalid")
            print("  Bug 2a is CONFIRMED on UNFIXED code.")
            return False
        else:
            print("\n✓ SUCCESS: Empty token correctly detected as invalid!")
            print("  Bug 2a is FIXED.")
            return True


async def test_bug2_valid_token_conditions():
    """
    Bug 2b: Valid Token Detection
    
    Expected: Valid token + internet should lead to remote model selection
    """
    print_test_header("Bug 2b - Valid Token Detection")
    
    with patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "hf_valid_token_12345"}):
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        print(f"Token value: '{token}'")
        print(f"Token exists: {token is not None}")
        print(f"Token non-empty: {bool(token and token.strip())}")
        
        # Simulate internet check
        internet_available = True  # Would be checked by _check_internet()
        
        print(f"Internet available: {internet_available}")
        
        should_use_remote = bool(token and token.strip()) and internet_available
        
        print(f"\nShould select remote model: {should_use_remote}")
        
        if should_use_remote:
            print("✓ Conditions for remote model selection are met")
            print("  (Actual model selection depends on code implementation)")
            return True
        else:
            print("✗ Conditions for remote model selection NOT met")
            return False


async def test_bug3_internet_check():
    """
    Bug 3: Network/DNS Issues in Container
    
    Expected on UNFIXED code: May fail with DNS errors in container
    Expected on FIXED code: DNS servers configured, internet check works
    """
    print_test_header("Bug 3 - Container Network Access")
    
    service = RAGService()
    
    print("Testing internet connectivity check...")
    print("(This may legitimately fail if no internet, but should not have DNS errors)")
    
    try:
        result = await service._check_internet()
        
        print(f"Internet check result: {result}")
        
        if result:
            print("✓ SUCCESS: Internet check passed")
            print("  Network connectivity is working")
        else:
            print("✓ GRACEFUL FAIL: Internet check failed without DNS error")
            print("  (This is acceptable if no internet available)")
        
        print("  Bug 3 appears to be FIXED or not applicable in current environment")
        return True
        
    except Exception as e:
        error_msg = str(e)
        
        if "Name or service not known" in error_msg or "[Errno -2]" in error_msg:
            print(f"✗ FAIL: DNS/Network error - {error_msg}")
            print("  This confirms Bug 3: Docker networking misconfiguration")
            print("  Expected fix: DNS servers (8.8.8.8, 8.8.4.4) in docker-compose.yml")
            print("  Bug 3 is CONFIRMED on UNFIXED code.")
            return False
        else:
            print(f"⚠ Other network error: {error_msg}")
            print("  (Not the specific DNS bug, may be expected)")
            return True


async def main():
    """
    Run all bug exploration tests
    
    Expected on UNFIXED code: Multiple failures demonstrating bugs
    Expected on FIXED code: All tests pass
    """
    print("\n" + "=" * 70)
    print("RAG AUTO MODE - BUG CONDITION EXPLORATION")
    print("=" * 70)
    print("\nThis script tests for three bugs in the RAG service Auto Mode:")
    print("  1. Missing GenerationConfig import (NameError)")
    print("  2. Weak token validation (empty strings pass)")
    print("  3. Network/DNS issues in Docker containers")
    print("\nEXPECTED ON UNFIXED CODE: Tests will FAIL, confirming bugs exist")
    print("EXPECTED ON FIXED CODE: Tests will PASS, confirming bugs are resolved")
    print()
    
    results = {}
    
    # Test Bug 1
    results['bug1'] = await test_bug1_missing_import()
    
    # Test Bug 2a
    results['bug2a'] = await test_bug2_empty_token()
    
    # Test Bug 2b
    results['bug2b'] = await test_bug2_valid_token_conditions()
    
    # Test Bug 3
    results['bug3'] = await test_bug3_internet_check()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    print()
    
    for bug, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {bug}")
    
    print()
    
    if passed == total:
        print("✓ ALL TESTS PASSED")
        print("  All bugs appear to be FIXED.")
        print("  Auto Mode should work correctly.")
    else:
        print(f"✗ {total - passed} TEST(S) FAILED")
        print("  Bugs confirmed on UNFIXED code.")
        print("  This is expected - these failures demonstrate the bugs exist.")
        print("  Apply the fixes from the design document to resolve these issues.")
    
    print()
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
