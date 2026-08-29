"""
Constants and configuration for dataset service.
"""

from pathlib import Path
from typing import FrozenSet

# Default configuration values
DEFAULT_DATASET_ID = "beir/quora"
DEFAULT_BATCH_SIZE = 1000
DEFAULT_VALIDATION_RETRIES = 3
DEFAULT_RETRY_DELAY = 10
MAX_RECURSION_DEPTH = 6
METADATA_FILE_SUFFIX = ".tmp"
DB_BATCH_PAGE_SIZE = 500

# File extensions considered as dataset data files
DATA_FILE_EXTENSIONS: FrozenSet[str] = frozenset({
    ".zip", ".gz", ".jsonl", ".json", ".qrels", 
    ".tsv", ".pklz4", ".npy", ".pkl", ".txt"
})
