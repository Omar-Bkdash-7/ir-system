from .core.config import PreprocessingConfig
from .core.pipeline import (
    TextPreprocessor,
    preprocess as _preprocess,
    preprocess_batch as _preprocess_batch,
    get_default_preprocessor,
    get_preprocessor,
)
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
    mode: Optional[str] = None,  # Kept for backward compatibility, currently unused
    config: Optional[Union[PreprocessingConfig, dict]] = None
) -> list[str]:
    """
    Preprocess text with optional configuration.

    Args:
        text: Text to preprocess
        mode: Reserved for future use (currently ignored for compatibility)
        config: PreprocessingConfig instance or dict

    Returns:
        List of preprocessed tokens
    """
    if isinstance(config, dict):
        config = PreprocessingConfig.from_dict(config)

    return _preprocess(text, config)


def preprocess_batch(
    texts: list[str],
    mode: Optional[str] = None,  # Kept for backward compatibility, currently unused
    config: Optional[Union[PreprocessingConfig, dict]] = None,
    batch_size: Optional[int] = None,
) -> list[list[str]]:
    """
    Preprocess multiple texts efficiently (single spaCy streaming pass).

    Args:
        texts: Texts to preprocess
        mode: Reserved for future use (currently ignored for compatibility)
        config: PreprocessingConfig instance or dict (shared for all texts)
        batch_size: Optional spaCy batching hint for nlp.pipe

    Returns:
        List of token lists, one per input text (order preserved, blanks -> [])
    """
    if isinstance(config, dict):
        config = PreprocessingConfig.from_dict(config)

    return _preprocess_batch(texts, config, batch_size=batch_size)


__all__ = [
    "PreprocessingConfig",
    "TextPreprocessor",
    "preprocess",
    "preprocess_batch",
    "get_default_preprocessor",
    "get_preprocessor",
    "warmup_resources",
    "ResourceNotFoundError",
    "TokenizationError",
]
