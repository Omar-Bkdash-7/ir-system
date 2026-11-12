import os
import logging

logger = logging.getLogger(__name__)

_device = None


def get_torch_device() -> str:
    global _device
    
    if _device is not None:
        return _device
    
    env_device = os.getenv("TORCH_DEVICE", "").lower().strip()
    
    if env_device in ("cpu", "cuda", "mps"):
        _device = env_device
        logger.info(f"Using device from TORCH_DEVICE env: {_device}")
        return _device
    
    try:
        import torch
        
        if torch.cuda.is_available():
            _device = "cuda"
            logger.info("CUDA available, using GPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = "mps"
            logger.info("MPS available, using Apple Silicon GPU")
        else:
            _device = "cpu"
            logger.info("No GPU available, using CPU")
    except ImportError:
        _device = "cpu"
        logger.info("PyTorch not available, defaulting to CPU")
    
    return _device

