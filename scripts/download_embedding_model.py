"""
Download and verify embedding model.
This script downloads the embedding model to the local cache directory.
"""
import sys
from pathlib import Path
from common.model_loader import load_embedding_model, get_embedding_model_config
from common.logging_config import setup_logging

logger = setup_logging("download_embedding_model")


def download_model():
    """Download and verify the embedding model."""
    
    logger.info("=" * 60)
    logger.info("Embedding Model Download & Verification")
    logger.info("=" * 60)
    
    model_config = get_embedding_model_config()
    
    logger.info("Model Configuration:")
    logger.info(f"  HuggingFace ID: {model_config['model_name']}")
    logger.info(f"  Local path: {model_config['local_path']}")
    logger.info(f"  Expected dimension: {model_config['dimension']}")
    logger.info(f"  Batch size: {model_config['batch_size']}")
    logger.info(f"  Normalize: {model_config['normalize']}")
    logger.info(f"  Device: {model_config['device']}")
    logger.info("")
    
    local_path = Path(model_config['local_path'])
    if local_path.exists():
        logger.info(f"Local model found at: {local_path}")
        logger.info("Model will be loaded from local path")
    else:
        logger.info(f"Local model not found at: {local_path}")
        logger.info("Model will be downloaded from HuggingFace")
    
    logger.info("")
    logger.info("Loading model...")
    
    try:
        model = load_embedding_model(cache=False)
        
        actual_dim = model.get_sentence_embedding_dimension()
        expected_dim = model_config['dimension']
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Model Loaded Successfully")
        logger.info("=" * 60)
        logger.info(f"Model dimension: {actual_dim}")
        logger.info(f"Device: {model.device}")
        
        if actual_dim != expected_dim:
            logger.warning("")
            logger.warning("⚠️  DIMENSION MISMATCH DETECTED")
            logger.warning(f"  Config expects: {expected_dim}")
            logger.warning(f"  Model actual: {actual_dim}")
            logger.warning("")
            logger.warning("  Please update configs/config.yaml:")
            logger.warning(f"    models.embedding.dimension: {actual_dim}")
        else:
            logger.info("Dimension matches configuration ✓")
        
        logger.info("")
        logger.info("Testing encoding...")
        
        test_texts = [
            "This is a test sentence.",
            "Another example for verification."
        ]
        
        embeddings = model.encode(
            test_texts,
            normalize_embeddings=model_config['normalize'],
            show_progress_bar=False
        )
        
        logger.info(f"Test encoding successful: shape={embeddings.shape}")
        logger.info("")
        logger.info("=" * 60)
        logger.info("Model is ready for use!")
        logger.info("=" * 60)
        
        if not local_path.exists():
            logger.info("")
            logger.info("💡 TIP: To use the model offline, copy it to:")
            logger.info(f"   {local_path.absolute()}")
            logger.info("")
            logger.info("   The model is currently cached in:")
            import os
            cache_dir = os.environ.get('SENTENCE_TRANSFORMERS_HOME', '~/.cache/sentence_transformers')
            logger.info(f"   {cache_dir}")
        
        return True
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 60)
        logger.error("Model Loading Failed")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("  1. Check internet connection (if downloading from HuggingFace)")
        logger.error("  2. Verify SENTENCE_TRANSFORMERS_HOME environment variable")
        logger.error("  3. Check disk space for model download")
        logger.error(f"  4. Ensure local model exists at: {local_path}")
        logger.error("")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
