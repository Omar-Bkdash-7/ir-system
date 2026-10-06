# Preservation Property Tests - Documentation

## Overview

This document describes the preservation property tests created for the RAG Auto Mode bugfixes. These tests verify that existing functionality is not broken when the three bugs are fixed.

## Test File

**Location**: `services/rag/test_preservation_properties.py`

**Purpose**: Capture and verify existing RAG service behavior that must be preserved after applying bugfixes.

## Requirements Coverage

The preservation tests validate all preservation requirements (3.1 - 3.9):

### Requirement 3.1: Auto Mode Fallback - No Token
**Tests**:
- `test_auto_mode_no_token_falls_back_to_local()`
- `test_property_auto_mode_without_token_always_selects_local()`

**Expected Behavior**: When Auto Mode is selected and no token is available (None), the system SHALL fallback to local model (DistilGPT2).

**Baseline**: This behavior should be observable on UNFIXED code and preserved on FIXED code.

### Requirement 3.2: Auto Mode Fallback - No Internet
**Tests**:
- `test_auto_mode_no_internet_falls_back_to_local()`

**Expected Behavior**: When Auto Mode is selected and internet is not available, the system SHALL fallback to local model (DistilGPT2).

**Baseline**: This behavior should be observable on UNFIXED code and preserved on FIXED code.

### Requirement 3.3: Explicit Remote Model Selection
**Tests**:
- `test_explicit_remote_model_selection_preserved()`
- `test_property_explicit_remote_selection_logic_preserved()`

**Expected Behavior**: When explicit remote model is requested with valid token and internet, the system SHALL use the remote model (FLAN-T5).

**Baseline**: This behavior should work on UNFIXED code (assuming token is valid) and remain unchanged on FIXED code.

### Requirement 3.4: Explicit Local Model Selection
**Tests**:
- `test_explicit_local_model_selection_preserved()`
- `test_property_explicit_local_always_honored()`

**Expected Behavior**: When explicit local model is requested, the system SHALL use the local model (DistilGPT2) regardless of token/internet availability.

**Baseline**: This behavior should work on UNFIXED code and remain unchanged on FIXED code.

### Requirement 3.5: Fallback Mechanism on Errors
**Tests**:
- `test_error_fallback_mechanism_preserved()`
- `test_exception_types_preserved()`

**Expected Behavior**: When any generation error occurs with the primary selected model, the system SHALL fallback to local model with appropriate error logging.

**Baseline**: Error handling should work on UNFIXED code and remain unchanged on FIXED code.

### Requirement 3.6: No Context Scenario
**Tests**:
- `test_no_context_response_preserved()`
- `test_context_retrieval_logic_preserved()`

**Expected Behavior**: When context retrieval returns no documents, the system SHALL return the "No relevant documents found" response.

**Baseline**: This behavior should work on UNFIXED code and remain unchanged on FIXED code.

### Requirement 3.7: API Response Structure
**Tests**:
- `test_api_response_structure_preserved()`
- `test_property_generation_params_passed_through()`

**Expected Behavior**: The `/generate` endpoint SHALL return a RAGResponse with all required fields. Generation parameters (temperature, max_tokens, use_context) SHALL continue to be passed correctly.

**Baseline**: API contract should be observable on UNFIXED code and preserved on FIXED code.

### Requirement 3.8: Health Endpoint Structure
**Tests**:
- `test_health_endpoint_structure_preserved()`

**Expected Behavior**: The `/health` endpoint SHALL return health status including token configuration status.

**Baseline**: Health endpoint structure should be observable on UNFIXED code and preserved on FIXED code.

### Requirement 3.9: Models Endpoint Structure
**Tests**:
- `test_models_endpoint_structure_preserved()`

**Expected Behavior**: The `/models` endpoint SHALL return available models information with the auto strategy description.

**Baseline**: Models endpoint structure should be observable on UNFIXED code and preserved on FIXED code.

## Property-Based Testing Strategy

The tests use **Hypothesis** for property-based testing, which generates many test cases automatically to provide stronger guarantees than manual unit tests alone.

### Test Strategies

1. **explicit_local_model_request**: Generates requests with explicit local model selection across various parameter combinations.

2. **explicit_remote_model_request**: Generates requests with explicit remote model selection across various parameter combinations.

3. **auto_mode_no_token_request**: Generates Auto Mode requests without token to test fallback behavior.

4. **generation_params_request**: Generates requests with various generation parameter combinations to verify they're passed through correctly.

### Property Tests

1. **test_property_auto_mode_without_token_always_selects_local**: 
   - Property: Auto Mode without token ALWAYS selects local model
   - Scope: ANY request configuration when token is absent
   - Examples: 10 generated test cases

2. **test_property_explicit_local_always_honored**:
   - Property: Explicit local model selection is ALWAYS honored
   - Scope: Regardless of token availability or internet connectivity
   - Test Cases: Multiple token states (None, "", "valid_token")

3. **test_property_explicit_remote_selection_logic_preserved**:
   - Property: Explicit remote model selection logic is preserved
   - Scope: When remote model is explicitly requested with valid token and internet
   - Examples: 10 generated test cases

4. **test_property_generation_params_passed_through**:
   - Property: Generation parameters are correctly passed through
   - Parameters: temperature, max_tokens, use_context, system_prompt
   - Examples: 15 generated test cases

5. **test_property_retrieval_params_honored**:
   - Property: Retrieval parameters are honored unchanged
   - Parameters: top_k, retrieval_model
   - Examples: 10 generated test cases

## Testing Approach

### Observation-First Methodology

The preservation tests follow an observation-first approach:

1. **Observe**: Run tests on UNFIXED code to establish baseline behavior
2. **Document**: Capture observed behavior in test assertions
3. **Verify**: Ensure tests PASS on unfixed code (confirms baseline)
4. **Preserve**: After applying fixes, tests should still PASS (confirms no regression)

### Expected Test Results

**On UNFIXED Code (Baseline)**:
- ✅ All preservation tests should **PASS**
- Confirms existing behavior is correctly captured
- Establishes baseline for regression detection

**On FIXED Code (After Bugfixes)**:
- ✅ All preservation tests should **STILL PASS**
- Confirms no regressions introduced
- Proves non-buggy code paths unchanged

**If Tests Fail After Fix**:
- ❌ Indicates regression
- Review the failing test to identify what changed
- Either fix the regression or update the test if behavior intentionally changed

## Mock Infrastructure

The tests use comprehensive mocking to isolate the code under test:

### AsyncMock Usage

- `service.retrieve_context`: Mocked to return controlled document sets
- `service._check_internet`: Mocked to control internet availability
- Environment variables: Patched using `patch.dict(os.environ, ...)`

### Test Isolation

Each test:
- Sets up isolated environment variables
- Mocks external dependencies (retrieval service, internet check)
- Tests only the model selection logic and request handling
- Does not require actual models to be available
- Does not require actual network connectivity

## Running the Tests

### Prerequisites

```bash
pip install pytest hypothesis pytest-asyncio
```

### Run All Preservation Tests

```bash
pytest services/rag/test_preservation_properties.py -v
```

### Run Specific Test

```bash
pytest services/rag/test_preservation_properties.py::TestPreservationProperties::test_auto_mode_no_token_falls_back_to_local -v
```

### Run with Coverage

```bash
pytest services/rag/test_preservation_properties.py --cov=services.rag.rag_service --cov-report=term-missing
```

### Run in Docker

```bash
docker-compose run --rm rag python -m pytest services/rag/test_preservation_properties.py -v
```

## Test Validation

A validation script is provided to verify test structure without running tests:

```bash
python services/rag/validate_preservation_tests.py
```

This script:
- ✅ Verifies all required tests are implemented
- ✅ Checks property-based testing integration
- ✅ Validates async test support
- ✅ Confirms mocking infrastructure
- ✅ Maps tests to requirements

## Integration with Bug Condition Tests

The preservation tests complement the bug condition exploration tests:

1. **Bug Condition Tests** (`test_bug_condition_exploration.py`):
   - Test inputs where `isBugCondition(input)` returns TRUE
   - FAIL on unfixed code (proves bugs exist)
   - PASS on fixed code (proves bugs are fixed)

2. **Preservation Tests** (`test_preservation_properties.py`):
   - Test inputs where `isBugCondition(input)` returns FALSE
   - PASS on unfixed code (baseline behavior)
   - PASS on fixed code (no regressions)

Together, they provide complete coverage:
- Bug condition tests validate the fix works
- Preservation tests validate nothing else broke

## Test Maintenance

### When to Update These Tests

Update preservation tests if:
- New preservation requirements are added
- Existing behavior intentionally changes (with user approval)
- New edge cases are discovered

Do NOT update these tests to:
- Make failing tests pass after a bugfix (that indicates regression)
- "Fix" tests that fail due to implementation changes (investigate first)

### Test Stability

These tests should be stable because:
- They test existing, documented behavior
- They use mocking to isolate dependencies
- They don't depend on external services or models
- They capture behavior that won't change with bugfixes

## Summary

The preservation property tests provide:
- ✅ Complete coverage of requirements 3.1 - 3.9
- ✅ Property-based testing for strong guarantees
- ✅ Observation-first methodology
- ✅ Comprehensive mocking infrastructure
- ✅ Clear documentation and validation tools
- ✅ Integration with bug condition tests

**Expected Outcome**: All tests PASS on both UNFIXED and FIXED code, proving no regressions.
