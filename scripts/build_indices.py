import sys
from pathlib import Path
from common.config import get_config
from common.logging_config import setup_logging

logger = setup_logging("build_indices")


def check_prerequisites():
    processed_path = Path(get_config("paths.processed_docs", "data/processed/quora_preprocessed.jsonl"))

    if not processed_path.exists():
        logger.error(f"Preprocessed documents not found at: {processed_path}")
        logger.error("Run: python scripts/preprocess_dataset.py")
        return False

    return True


def build_all_indices():
    logger.info("Building all indices")

    if not check_prerequisites():
        sys.exit(1)

    logger.info("Step 1/2: Building TF-IDF Index")
    try:
        from scripts.build_tfidf_index import build_tfidf_index
        build_tfidf_index(force_rebuild=False)
        logger.info("TF-IDF Index completed")
    except Exception as e:
        logger.error(f"TF-IDF Index failed: {e}")
        sys.exit(1)

    logger.info("Step 2/2: Building Embedding Index")
    try:
        from services.indexing.embedding_indexer import EmbeddingIndexer
        indexer = EmbeddingIndexer()
        if indexer.exists():
            logger.info("Embedding index already exists, skipping")
        else:
            indexer.build()
            indexer.save()
            logger.info("Embedding Index completed")
    except Exception as e:
        logger.error(f"Embedding Index failed: {e}")
        sys.exit(1)

    logger.info("All indices built successfully")


if __name__ == "__main__":
    build_all_indices()
