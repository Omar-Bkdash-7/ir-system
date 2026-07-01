from .core.config import PreprocessingConfig
from .core.pipeline import TextPreprocessor, preprocess as _preprocess, get_default_preprocessor
from .core.resources import warmup_resources
from .core.exceptions import (
    ResourceNotFoundError,
    TokenizationError,
)
from typing import Optional, Union


def preprocess(
    text: str, 
    mode: Optional[str] = None,
    config: Optional[Union[PreprocessingConfig, dict]] = None
) -> list[str]:
    if isinstance(config, dict):
        config = PreprocessingConfig(**config)
    
    return _preprocess(text, config)


__all__ = [
    "PreprocessingConfig",
    "TextPreprocessor",
    "preprocess",
    "get_default_preprocessor",
    "warmup_resources",
    "ResourceNotFoundError",
    "TokenizationError",
]
