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
