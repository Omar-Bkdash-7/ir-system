# Dataset Management Scripts

This directory contains maintenance scripts for dataset storage management.

## Available Scripts

### 1. Fix Dataset Storage
**File**: `fix_dataset_storage.py`

**Purpose**: Consolidates datasets to the correct storage location and removes duplicates.

**Usage**:
```bash
python scripts/fix_dataset_storage.py
```

**What it does**:
- Scans for datasets in all locations
- Identifies duplicate storage directories
- Moves datasets to correct location (`.ir_datasets`)
- Removes empty duplicate directories
- Provides detailed logging of all operations

**When to use**:
- After manual dataset downloads
- When duplicate locations exist
- Storage structure appears incorrect
- Before system deployment

**Example output**:
```
================================================================================
Dataset Storage Consolidation Script
================================================================================

Scanning correct location: data/datasets/.ir_datasets
Found dataset 'beir/quora' at data/datasets/.ir_datasets/beir/quora (74.5 MB)

Scanning duplicate location: data/datasets/ir_datasets
No duplicates found.

✅ Dataset storage structure has been fixed!
All datasets are now in: data/datasets/.ir_datasets/
```

---

### 2. Validate Dataset Structure
**File**: `validate_dataset_structure.py`

**Purpose**: Validates the dataset storage structure and configuration.

**Usage**:
```bash
python scripts/validate_dataset_structure.py
```

**What it checks**:
1. Directory structure correctness
2. Downloaded datasets detection
3. metadata.json consistency
4. Environment variable configuration
5. Indices directory organization

**When to use**:
- Regular system health checks
- After storage changes
- Before building indices
- Troubleshooting storage issues
- After running fix script

**Example output**:
```
================================================================================
Dataset Storage Structure Validation
================================================================================

1. Checking directory structure...
   ✅ data/datasets/ exists
   ✅ data/datasets/.ir_datasets/ exists (correct location)
   ✅ No duplicate ir_datasets directory

2. Checking for downloaded datasets...
   ✅ Found 1 dataset(s):
      - beir/quora (74.5 MB)

3. Checking metadata.json...
   ✅ metadata.json exists
      Active dataset: beir/quora
      Status: success

4. Checking environment configuration...
   ✅ IR_DATASETS_HOME is set in .env

5. Checking indices directory...
   ℹ️  No indices built yet

================================================================================
Validation Summary
================================================================================

✅ All checks passed! Dataset storage structure is correct.
```

---

## Workflow Examples

### Regular Maintenance
```bash
# Check system health
python scripts/validate_dataset_structure.py

# If issues found, fix them
python scripts/fix_dataset_storage.py

# Verify fix was successful
python scripts/validate_dataset_structure.py
```

### After Manual Changes
```bash
# Always validate after manual file operations
python scripts/validate_dataset_structure.py

# Fix if needed
python scripts/fix_dataset_storage.py
```

### Before Deployment
```bash
# Ensure clean state
python scripts/fix_dataset_storage.py
python scripts/validate_dataset_structure.py
```

### CI/CD Integration
```bash
# Add to CI pipeline
python scripts/validate_dataset_structure.py || exit 1
```

---

## Script Requirements

### Dependencies
- Python 3.8+
- No external packages required (uses stdlib only)
- Works on Windows, Linux, macOS

### File Structure
Both scripts expect this directory structure:
```
project_root/
├── data/
│   ├── datasets/
│   │   └── .ir_datasets/
│   └── metadata.json
├── scripts/
│   ├── fix_dataset_storage.py
│   └── validate_dataset_structure.py
└── .env
```

---

## Exit Codes

### Validation Script
- `0`: All checks passed
- `1`: Issues found or validation failed

### Fix Script
- `0`: Successfully fixed issues
- `1`: Fix failed with error

---

## Logging

Both scripts use Python's `logging` module:
- `INFO`: Normal operations
- `WARNING`: Non-critical issues
- `ERROR`: Critical problems

**Customize logging**:
```python
# In script, modify:
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for verbose output
    format='%(levelname)s: %(message)s'
)
```

---

## Troubleshooting

### Script won't run
**Check**:
- Python version: `python --version`
- Working directory: Should be project root
- File permissions: Scripts should be readable

### Permission errors
**Solution**:
```bash
# Windows (as Administrator)
python scripts/fix_dataset_storage.py

# Linux/Mac
sudo python scripts/fix_dataset_storage.py
```

### Path errors
**Ensure**:
- Running from project root directory
- Relative paths in scripts are correct
- `data/` directory exists

---

## Best Practices

### ✅ Do
- Run validation regularly
- Fix issues immediately
- Review script output
- Keep backups before fixing
- Test in development first

### ❌ Don't
- Run fix script without validation first
- Modify scripts without understanding
- Ignore warning messages
- Delete datasets manually
- Skip validation after fixes

---

## Additional Resources

### Documentation
- `../data/README_DATASETS.txt` - Storage structure guide
- `../DATASET_STORAGE_FIX.md` - Detailed fix documentation
- `../FINAL_REPORT.md` - Complete project report
- `../docs/dataset_storage_ar.md` - Arabic documentation

### Related Files
- `../frontend/app_pages/indexing.py` - UI that uses datasets
- `../services/indexing/dataset/` - Dataset service modules
- `../.gitignore` - Storage exclusion rules

---

## Support

### Questions?
1. Read `../FINAL_REPORT.md` for complete information
2. Check `../data/README_DATASETS.txt` for structure details
3. Review script source code (both are well-commented)

### Found a bug?
1. Run validation to confirm issue
2. Check script output logs
3. Verify file permissions
4. Review directory structure

---

**Last Updated**: August 1, 2026  
**Scripts Version**: 1.0  
**Status**: Production Ready
