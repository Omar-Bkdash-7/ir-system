from services.indexing.indexers import TFIDFIndexer, EmbeddingIndexer, BM25Indexer
import logging

logger = logging.getLogger(__name__)


def build_tfidf_index(force_rebuild: bool = False) -> dict:
    logger.info(f"Building TF-IDF index (force_rebuild={force_rebuild})")

    indexer = TFIDFIndexer()

    if indexer.exists() and not force_rebuild:
        logger.info("TF-IDF index already exists, skipping build")
        return {
            "status": "skipped",
            "message": "TF-IDF index already exists. Use force_rebuild=true to rebuild."
        }

    try:
        indexer.build()
        indexer.save()

        logger.info("TF-IDF index built successfully")

        return {
            "status": "success",
            "vocab_size": len(indexer.vectorizer.vocabulary_),
            "matrix_shape": list(indexer.matrix.shape),
            "total_docs": len(indexer.doc_ids)
        }

    except Exception as e:
        logger.error(f"Failed to build TF-IDF index: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


def build_embedding_index(force_rebuild: bool = False) -> dict:
    logger.info(f"Building Embedding index (force_rebuild={force_rebuild})")

    indexer = EmbeddingIndexer()

    if indexer.exists() and not force_rebuild:
        logger.info("Embedding index already exists, skipping build")
        return {
            "status": "skipped",
            "message": "Embedding index already exists. Use force_rebuild=true to rebuild."
        }

    try:
        indexer.build()
        indexer.save()

        logger.info("Embedding index built successfully")

        return {
            "status": "success",
            "shape": list(indexer.embeddings.shape),
            "total_docs": len(indexer.doc_ids),
            "memory_mb": round(indexer.embeddings.nbytes / (1024**2), 2),
            "device": indexer.device
        }

    except Exception as e:
        logger.error(f"Failed to build Embedding index: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


def build_bm25_index(force_rebuild: bool = False) -> dict:
    logger.info(f"Building BM25 index (force_rebuild={force_rebuild})")

    indexer = BM25Indexer()

    if indexer.exists() and not force_rebuild:
        logger.info("BM25 index already exists, skipping build")
        return {
            "status": "skipped",
            "message": "BM25 index already exists. Use force_rebuild=true to rebuild."
        }

    try:
        indexer.build()
        indexer.save()

        logger.info("BM25 index built successfully")

        return {
            "status": "success",
            "vocab_size": len(indexer.inverted_index),
            "total_docs": indexer.total_docs,
            "avg_doc_length": round(indexer.avg_doc_length, 2)
        }

    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }
