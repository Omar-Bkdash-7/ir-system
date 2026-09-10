from .core.config import PreprocessingConfig
from .core.pipeline import TextPreprocessor, preprocess as _preprocess, get_default_preprocessor
from .core.resources import warmup_resources
from .core.exceptions import (
    ResourceNotFoundError,
    TokenizationError,
)
from typing import Optional, Union
import logging

logger = logging.getLogger("services.preprocessing.preprocessing_service")


def preprocess(
    text: str, 
    mode: Optional[str] = None,
    config: Optional[Union[PreprocessingConfig, dict]] = None
) -> list[str]:
    if isinstance(config, dict):
        logger.warning(f"[PREPROCESSING_SERVICE] Converting dict to PreprocessingConfig: use_stemming={config.get('use_stemming')}, use_lemmatization={config.get('use_lemmatization')}")
        config = PreprocessingConfig.from_dict(config)
        logger.warning(f"[PREPROCESSING_SERVICE] After conversion: use_stemming={config.use_stemming}, use_lemmatization={config.use_lemmatization}")
    
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
