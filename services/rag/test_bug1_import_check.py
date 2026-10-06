"""
Bug 1: Simple Import Check

This script checks if GenerationConfig is imported in rag_service.py
without needing to run the full code or install dependencies.

Expected on UNFIXED code: "GenerationConfig" NOT found in imports
Expected on FIXED code: "GenerationConfig" found in imports
"""

import re
from pathlib import Path


def check_generationconfig_import():
    """Check if GenerationConfig is imported in rag_service.py"""
    
    print("=" * 70)
    print("BUG 1: GENERATIONCONFIG IMPORT CHECK")
    print("=" * 70)
    print()
    
    # Read rag_service.py
    rag_service_path = Path(__file__).parent / "rag_service.py"
    
    with open(rag_service_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Look for GenerationConfig in imports (first 20 lines)
    import_section = '\n'.join(lines[:20])
    
    print("Checking import section of rag_service.py...")
    print()
    
    # Pattern 1: Direct import from schemas.rag_schemas
    pattern1 = r'from\s+services\.rag\.schemas\.rag_schemas\s+import.*GenerationConfig'
    
    # Pattern 2: Import from schemas __init__ (would work too)
    pattern2 = r'from\s+services\.rag\.schemas\s+import.*GenerationConfig'
    
    has_direct_import = bool(re.search(pattern1, import_section))
    has_init_import = bool(re.search(pattern2, import_section))
    
    print("Import section (first 20 lines):")
    print("-" * 70)
    for i, line in enumerate(lines[:20], 1):
        if line.strip():
            print(f"{i:3d}: {line}")
    print("-" * 70)
    print()
    
    print("Checks:")
    print(f"  - Direct import from rag_schemas: {has_direct_import}")
    print(f"  - Import from schemas __init__: {has_init_import}")
    print()
    
    # Check if GenerationConfig is used in the code
    usage_pattern = r'\bGenerationConfig\s*\('
    usages = re.findall(usage_pattern, content)
    
    print(f"GenerationConfig instantiation count: {len(usages)}")
    
    # Find line numbers where it's used
    for i, line in enumerate(lines, 1):
        if re.search(usage_pattern, line):
            print(f"  Line {i}: {line.strip()}")
    
    print()
    
    # Verdict
    if not (has_direct_import or has_init_import):
        print("✗ FAIL: GenerationConfig is NOT imported!")
        print("  Bug 1 CONFIRMED on UNFIXED code.")
        print("  The code uses GenerationConfig but doesn't import it.")
        print("  This will cause: NameError: name 'GenerationConfig' is not defined")
        print()
        print("FIX: Add this import to rag_service.py:")
        print("  from services.rag.schemas.rag_schemas import GenerationConfig")
        return False
    else:
        print("✓ PASS: GenerationConfig is properly imported!")
        print("  Bug 1 is FIXED.")
        return True


def check_token_validation():
    """Check token validation logic"""
    
    print()
    print("=" * 70)
    print("BUG 2: TOKEN VALIDATION CHECK")
    print("=" * 70)
    print()
    
    rag_service_path = Path(__file__).parent / "rag_service.py"
    
    with open(rag_service_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Find the token validation line (around line 59)
    print("Looking for token validation logic...")
    print()
    
    for i, line in enumerate(lines, 1):
        if 'if hf_token and' in line and '_check_internet' in line:
            print(f"Found at line {i}:")
            print(f"  {line.strip()}")
            print()
            
            # Check if it has .strip() validation
            if 'hf_token.strip()' in line:
                print("✓ PASS: Token validation includes .strip() check")
                print("  Bug 2 is FIXED.")
                return True
            else:
                print("✗ FAIL: Token validation is weak!")
                print("  Bug 2 CONFIRMED on UNFIXED code.")
                print("  Current: if hf_token and await self._check_internet():")
                print("  Problem: Empty string '' passes `if hf_token` check")
                print()
                print("FIX: Change to:")
                print("  if hf_token and hf_token.strip() and await self._check_internet():")
                return False
    
    print("⚠ Could not find token validation line")
    return None


def check_internet_timeout():
    """Check internet connectivity timeout"""
    
    print()
    print("=" * 70)
    print("BUG 2 (PART 2): INTERNET CHECK TIMEOUT")
    print("=" * 70)
    print()
    
    rag_service_path = Path(__file__).parent / "rag_service.py"
    
    with open(rag_service_path, 'r') as f:
        lines = f.readlines()
    
    # Find _check_internet method
    in_method = False
    for i, line in enumerate(lines, 1):
        if 'def _check_internet' in line:
            in_method = True
            start_line = i
        
        if in_method and 'timeout=' in line:
            print(f"Found at line {i}:")
            print(f"  {line.strip()}")
            print()
            
            # Extract timeout value
            timeout_match = re.search(r'timeout=(\d+\.?\d*)', line)
            if timeout_match:
                timeout = float(timeout_match.group(1))
                print(f"Current timeout: {timeout} seconds")
                
                if timeout >= 10.0:
                    print("✓ PASS: Timeout is adequate for Docker networking")
                    print("  Bug 2 (internet check) is FIXED.")
                    return True
                else:
                    print("⚠ WARNING: Timeout may be too short for Docker")
                    print("  Current: 5.0 seconds")
                    print("  Recommended: 10.0 seconds for container networking")
                    print()
                    print("FIX: Increase timeout:")
                    print("  async with httpx.AsyncClient(timeout=10.0) as client:")
                    return False
            break
        
        if in_method and line.strip() == '':
            in_method = False
    
    print("⚠ Could not find timeout in _check_internet method")
    return None


def main():
    """Run all static code checks"""
    
    print()
    print("=" * 70)
    print("RAG AUTO MODE - STATIC BUG ANALYSIS")
    print("=" * 70)
    print()
    print("This script analyzes the code without running it.")
    print("It checks for the three bugs in RAG Auto Mode.")
    print()
    
    results = {}
    
    # Check Bug 1
    results['bug1_import'] = check_generationconfig_import()
    
    # Check Bug 2a
    results['bug2_token'] = check_token_validation()
    
    # Check Bug 2b
    results['bug2_timeout'] = check_internet_timeout()
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    passed = sum(1 for v in results.values() if v is True)
    total = len([v for v in results.values() if v is not None])
    
    print(f"Results: {passed}/{total} checks passed")
    print()
    
    for check, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"  {status} - {check}")
    
    print()
    
    if passed == total and total > 0:
        print("✓ ALL CHECKS PASSED")
        print("  All bugs appear to be FIXED.")
    else:
        print(f"✗ {total - passed}/{total} CHECK(S) FAILED")
        print("  Bugs confirmed on UNFIXED code.")
        print("  Apply the fixes shown above to resolve the issues.")
    
    print()
    
    # Note about Bug 3
    print("=" * 70)
    print("NOTE: Bug 3 (Docker Networking)")
    print("=" * 70)
    print()
    print("Bug 3 cannot be checked with static analysis.")
    print("It requires Docker configuration changes in docker-compose.yml:")
    print()
    print("Add DNS configuration to the RAG service:")
    print("  dns:")
    print("    - 8.8.8.8")
    print("    - 8.8.4.4")
    print()
    print("This enables DNS resolution and internet access from containers.")
    print()


if __name__ == "__main__":
    main()
