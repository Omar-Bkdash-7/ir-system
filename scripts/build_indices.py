import sys
from pathlib import Path
from common.config import get_config
from common.logging_config import setup_logging

logger = setup_logging("build_indices")


def check_prerequisites() -> bool:
    processed_path = Path(get_config("paths.processed_docs", "data/processed/quora_preprocessed.jsonl"))

    if not processed_path.exists():
        logger.error("=" * 60)
        logger.error("PREREQUISITE MISSING")
        logger.error("=" * 60)
        logger.error(f"Preprocessed documents not found at: {processed_path}")
        logger.error("")
        logger.error("Please run preprocessing first:")
        logger.error("  python scripts/preprocess_dataset.py")
        logger.error("=" * 60)
        return False

    logger.info(f"Preprocessed documents found: {processed_path}")
    return True


def build_tfidf() -> bool:
    logger.info("=" * 60)
    logger.info("STEP 1/3: Building TF-IDF Index")
    logger.info("=" * 60)

    try:
        from scripts.build_tfidf_index import build_tfidf_index
        build_tfidf_index(force_rebuild=False)
        logger.info("✓ TF-IDF Index completed successfully")
        return True

    except Exception as e:
        logger.error(f"✗ TF-IDF Index failed: {e}", exc_info=True)
        return False


def build_bm25() -> bool:
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2/3: Building BM25 Index")
    logger.info("=" * 60)

    try:
        from services.indexing.bm25_indexer import BM25Indexer

        indexer = BM25Indexer()

        if indexer.exists():
            logger.info("BM25 index already exists")
            logger.info("Skipping build (use --force to rebuild)")
            return True

        logger.info("Starting BM25 index build...")

        indexer.build()
        indexer.save()

        logger.info("✓ BM25 Index completed successfully")
        return True

    except Exception as e:
        logger.error(f"✗ BM25 Index failed: {e}", exc_info=True)
        return False


def build_embedding() -> bool:
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3/3: Building Embedding Index")
    logger.info("=" * 60)

    try:
        from services.indexing.embedding_indexer import EmbeddingIndexer

        indexer = EmbeddingIndexer()

        if indexer.exists():
            logger.info("Embedding index already exists")
            logger.info("Skipping build (use --force to rebuild)")
            return True

        logger.info("Starting embedding index build...")
        logger.info(f"Device: {indexer.device}")

        indexer.build()
        indexer.save()

        logger.info("✓ Embedding Index completed successfully")
        return True

    except Exception as e:
        logger.error(f"✗ Embedding Index failed: {e}", exc_info=True)
        return False


def build_all_indices(force_rebuild: bool = False):
    logger.info("")
    logger.info("=" * 60)
    logger.info("BUILDING ALL INDICES")
    logger.info("=" * 60)
    logger.info("")

    if not check_prerequisites():
        sys.exit(1)

    logger.info("")

    tfidf_success = build_tfidf()

    if not tfidf_success:
        logger.error("")
        logger.error("=" * 60)
        logger.error("BUILD FAILED - TF-IDF index failed")
        logger.error("=" * 60)
        sys.exit(1)

    bm25_success = build_bm25()

    embedding_success = build_embedding()

    logger.info("")
    logger.info("=" * 60)

    if tfidf_success and bm25_success and embedding_success:
        logger.info("✓ ALL INDICES BUILT SUCCESSFULLY")
    elif tfidf_success and bm25_success:
        logger.warning("⚠ PARTIAL SUCCESS - Embedding index failed")
    elif tfidf_success:
        logger.warning("⚠ PARTIAL SUCCESS - BM25 or Embedding index failed")
    else:
        logger.error("✗ BUILD FAILED")

    logger.info("=" * 60)
    logger.info("")

    if not (tfidf_success and bm25_success and embedding_success):
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build all indices (TF-IDF, BM25, and Embedding)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if indices already exist"
    )
    args = parser.parse_args()

    build_all_indices(force_rebuild=args.force)
