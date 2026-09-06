import sys
import argparse
from pathlib import Path
from services.indexing.embedding_indexer import EmbeddingIndexer
from common.config import get_config
from common.logging_config import setup_logging

logger = setup_logging("build_embedding_index")


def build_embedding_index(force_rebuild: bool = False):
    processed_path = Path(get_config("paths.processed_docs", "data/processed/quora_preprocessed.jsonl"))

    if not processed_path.exists():
        logger.error(f"Preprocessed documents not found at {processed_path}")
        logger.error("Run: python scripts/preprocess_dataset.py")
        sys.exit(1)

    indexer = EmbeddingIndexer()

    if indexer.exists() and not force_rebuild:
        logger.info("Embedding index already exists")
        logger.info("Use --force to rebuild")

        indexer.load()
        logger.info("Current index stats:")
        logger.info(f"  Documents: {len(indexer.doc_ids)}")
        logger.info(f"  Embedding dimension: {indexer.embeddings.shape[1]}")
        logger.info(f"  Shape: {indexer.embeddings.shape}")
        logger.info(f"  Memory: {indexer.embeddings.nbytes / (1024**2):.2f} MB")
        return

    logger.info("=" * 60)
    logger.info("Building Embedding Index")
    logger.info("=" * 60)
    logger.info(f"Device: {indexer.device}")
    
    from common.model_loader import get_embedding_model_config
    model_config = get_embedding_model_config()
    logger.info(f"Model: {model_config['model_name']}")
    logger.info(f"Local path: {model_config['local_path']}")
    logger.info("")

    try:
        indexer.build()
        indexer.save()

        logger.info("")
        logger.info("=" * 60)
        logger.info("Embedding Index Built Successfully")
        logger.info("=" * 60)
        logger.info(f"Documents indexed: {len(indexer.doc_ids)}")
        logger.info(f"Embedding dimension: {indexer.embeddings.shape[1]}")
        logger.info(f"Shape: {indexer.embeddings.shape}")
        logger.info(f"Memory: {indexer.embeddings.nbytes / (1024**2):.2f} MB")
        logger.info(f"Device used: {indexer.device}")
        logger.info("=" * 60)
        logger.info("Output files:")
        logger.info(f"  - {indexer.embeddings_path}")
        logger.info(f"  - {indexer.doc_id_map_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to build Embedding index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Embedding index from preprocessed documents")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()

    build_embedding_index(force_rebuild=args.force)
