import yaml
from pathlib import Path
from typing import Any

_config_cache = None

def load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache

def get_config(key_path: str, default: Any = None) -> Any:
    config = load_config()
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

def get_dataset_specific_path(key_path: str, dataset_name: str, default: Any = None) -> str:
    """
    Get config path with {dataset} placeholder substituted.
    
    Args:
        key_path: Config key path (e.g., 'indices.bm25_index')
        dataset_name: Dataset name (e.g., 'beir/quora')
        default: Default value if key not found
    
    Returns:
        Path with {dataset} replaced by dataset slug (e.g., 'beir_quora')
    """
    path_template = get_config(key_path, default)
    if path_template and isinstance(path_template, str):
        dataset_slug = dataset_name.replace('/', '_').replace('-', '_').replace('.', '_')
        return path_template.format(dataset=dataset_slug)
    return path_template


def get_active_dataset_path(key_path: str, default: Any = None) -> str:
    """
    Get config path for the active dataset.
    
    Args:
        key_path: Config key path (e.g., 'indices.bm25_index')
        default: Default value if key not found
    
    Returns:
        Path with {dataset} replaced by active dataset slug
        
    Raises:
        ValueError: If no active dataset is set
    """
    try:
        from services.indexing.dataset_service import get_metadata
        metadata = get_metadata()
        dataset_name = metadata.get("dataset_name")
        
        if not dataset_name:
            raise ValueError("No active dataset found in metadata")
        
        return get_dataset_specific_path(key_path, dataset_name, default)
    except ImportError:
        # Fallback for services that don't have access to indexing service
        # This should use the legacy path format
        path_template = get_config(key_path, default)
        if path_template and isinstance(path_template, str) and '{dataset}' in path_template:
            raise ValueError(
                "Cannot resolve dataset-specific path: no active dataset available. "
                "Please select a dataset first."
            )
        return path_template


