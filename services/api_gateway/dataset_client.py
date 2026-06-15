import json
from pathlib import Path
from common.config import get_config

def get_dataset_info() -> dict:
    metadata_path = Path(get_config("paths.metadata", "data/metadata.json"))
    if not metadata_path.exists():
        raise FileNotFoundError("Dataset metadata not found")
    
    with open(metadata_path) as f:
        return json.load(f)
