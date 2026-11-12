import sys
import argparse
from pathlib import Path
from services.indexing.tfidf_indexer import TFIDFIndexer
from common.config import get_config
from common.logging_config import setup_logging

logger = setup_logging("build_tfidf_index")


def build_tfidf_index(force_rebuild: bool = False):
    processed_path = Path(get_config("paths.processed_docs", "data/processed/quora_preprocessed.jsonl"))

    if not processed_path.exists():
        logger.error(f"Preprocessed documents not found at {processed_path}")
        logger.error("Run: python scripts/preprocess_dataset.py")
        sys.exit(1)

    indexer = TFIDFIndexer()

    if indexer.exists() and not force_rebuild:
        logger.info("TF-IDF index already exists")
        logger.info("Use --force to rebuild")
        indexer.load()
        logger.info("Current index stats:")
        logger.info(f"  Documents: {len(indexer.doc_ids)}")
        logger.info(f"  Vocabulary size: {len(indexer.vectorizer.vocabulary_)}")
        logger.info(f"  Matrix shape: {indexer.matrix.shape}")
        return

    logger.info("Building TF-IDF Index")

    try:
        indexer.build()
        indexer.save()

        logger.info("TF-IDF Index built successfully")
        logger.info(f"Documents indexed: {len(indexer.doc_ids)}")
        logger.info(f"Vocabulary size: {len(indexer.vectorizer.vocabulary_)}")
        logger.info(f"Matrix shape: {indexer.matrix.shape}")
        logger.info(f"Matrix density: {indexer.matrix.nnz / (indexer.matrix.shape[0] * indexer.matrix.shape[1]):.4%}")
        logger.info("Output files:")
        logger.info(f"  {indexer.vectorizer_path}")
        logger.info(f"  {indexer.matrix_path}")
        logger.info(f"  {indexer.doc_ids_path}")

    except Exception as e:
        logger.error(f"Failed to build TF-IDF index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TF-IDF index from preprocessed documents")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()

    build_tfidf_index(force_rebuild=args.force)
