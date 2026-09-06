"""
Centralized model loading utilities for embedding models.
Provides consistent model loading across all services.
"""
import os
import logging
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
from common.config import get_config
from common.device import get_torch_device
from common.exceptions import ModelLoadError

logger = logging.getLogger(__name__)

_embedding_model_cache = None


def get_embedding_model_path() -> str:
    """
    Get the embedding model path with local fallback.
    
    Returns:
        str: Model identifier or local path
    """
    local_path = get_config("models.embedding.local_path", "data/models/all-MiniLM-L6-v2")
    model_name = get_config("models.embedding.model_name", "sentence-transformers/all-MiniLM-L6-v2")
    
    if Path(local_path).exists():
        logger.info(f"Using local embedding model from: {local_path}")
        return local_path
    
    logger.info(f"Using HuggingFace model: {model_name}")
    return model_name


def load_embedding_model(device: Optional[str] = None, cache: bool = True) -> SentenceTransformer:
    """
    Load embedding model with caching and local fallback.
    
    Args:
        device: Device to load model on (default: auto-detected)
        cache: Whether to use cached model instance (default: True)
    
    Returns:
        SentenceTransformer: Loaded model
        
    Raises:
        ModelLoadError: If model loading fails
    """
    global _embedding_model_cache
    
    if cache and _embedding_model_cache is not None:
        logger.debug("Using cached embedding model")
        return _embedding_model_cache
    
    if device is None:
        device = get_torch_device()
    
    model_path = get_embedding_model_path()
    trust_remote_code = get_config("models.embedding.trust_remote_code", False)
    
    logger.info(f"Loading embedding model: {model_path} on device: {device}")
    
    try:
        model = SentenceTransformer(
            model_path,
            device=device,
            trust_remote_code=trust_remote_code
        )
        
        expected_dim = get_config("models.embedding.dimension", 384)
        actual_dim = model.get_sentence_embedding_dimension()
        
        if actual_dim != expected_dim:
            logger.warning(
                f"Model dimension mismatch: config={expected_dim}, actual={actual_dim}. "
                f"Consider updating config.yaml to dimension: {actual_dim}"
            )
        
        logger.info(
            f"Embedding model loaded successfully: "
            f"dimension={actual_dim}, device={device}"
        )
        
        if cache:
            _embedding_model_cache = model
        
        return model
        
    except Exception as e:
        error_msg = (
            f"Failed to load embedding model '{model_path}'. "
            f"Check: (1) internet connection for HuggingFace download, "
            f"(2) local model exists at '{get_config('models.embedding.local_path')}', "
            f"(3) SENTENCE_TRANSFORMERS_HOME is set correctly. "
            f"Original error: {str(e)}"
        )
        logger.error(error_msg)
        raise ModelLoadError(error_msg)


def get_embedding_model_config() -> dict:
    """
    Get embedding model configuration.
    
    Returns:
        dict: Model configuration
    """
    return {
        "model_name": get_config("models.embedding.model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        "local_path": get_config("models.embedding.local_path", "data/models/all-MiniLM-L6-v2"),
        "dimension": get_config("models.embedding.dimension", 384),
        "batch_size": get_config("models.embedding.batch_size", 64),
        "normalize": get_config("models.embedding.normalize", True),
        "trust_remote_code": get_config("models.embedding.trust_remote_code", False),
        "device": get_torch_device()
    }


def clear_embedding_model_cache():
    """Clear the cached embedding model instance."""
    global _embedding_model_cache
    _embedding_model_cache = None
    logger.info("Embedding model cache cleared")
