import logging
import sys
from pathlib import Path
from common.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    processed_path = Path(get_config("paths.processed_docs", "data/processed/quora_preprocessed.jsonl"))
    
    if not processed_path.exists():
        logger.error("=" * 60)
        logger.error("PREREQUISITE MISSING")
        logger.error("=" * 60)
        logger.error(f"Preprocessed documents not found at: {processed_path}")
        logger.error("")
        logger.error("Please run preprocessing first:")
        logger.error("  docker compose run --rm preprocessing python scripts/preprocess_dataset.py")
        logger.error("=" * 60)
        return False
    
    return True

def build_all_indices():
    logger.info("=" * 60)
    logger.info("BUILDING ALL INDICES")
    logger.info("=" * 60)
    
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 1: Build TF-IDF Index
    logger.info("")
    logger.info("Step 1/3: Building TF-IDF Index...")
    logger.info("-" * 60)
    try:
        from scripts.build_tfidf_index import build_tfidf_index
        build_tfidf_index(force_rebuild=False)
        logger.info("✓ TF-IDF Index completed")
    except Exception as e:
        logger.error(f"✗ TF-IDF Index failed: {str(e)}")
        sys.exit(1)
    
    # Step 2: Build BM25 Index (placeholder for now)
    logger.info("")
    logger.info("Step 2/3: Building BM25 Index...")
    logger.info("-" * 60)
    logger.info("⊙ BM25 Index not implemented yet")
    
    # Step 3: Build Embedding Index (placeholder for now)
    logger.info("")
    logger.info("Step 3/3: Building Embedding Index...")
    logger.info("-" * 60)
    logger.info("⊙ Embedding Index not implemented yet")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("INDEX BUILDING COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    build_all_indices()
