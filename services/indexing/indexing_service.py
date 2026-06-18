from services.indexing.tfidf_indexer import TFIDFIndexer
from services.indexing.embedding_indexer import EmbeddingIndexer
from services.indexing.dataset_service import get_metadata
import logging

logger = logging.getLogger(__name__)


def get_index_stats() -> dict:
    try:
        metadata = get_metadata()
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        metadata = {"doc_count": "unknown"}

    tfidf_stats = _get_tfidf_stats()
    embedding_stats = _get_embedding_stats()

    return {
        "dataset": {
            "total_docs": metadata.get("doc_count", "unknown"),
            "name": metadata.get("dataset_name", "unknown")
        },
        "indices": {
            "tfidf": tfidf_stats,
            "embedding": embedding_stats
        }
    }


def _get_tfidf_stats() -> dict:
    try:
        indexer = TFIDFIndexer()
        if not indexer.exists():
            return {"status": "not_built"}

        indexer.load()
        return {
            "status": "ready",
            "vocab_size": len(indexer.vectorizer.vocabulary_),
            "matrix_shape": list(indexer.matrix.shape),
            "total_docs": len(indexer.doc_ids)
        }
    except Exception as e:
        logger.error(f"Failed to get TF-IDF stats: {e}")
        return {"status": "error", "message": str(e)}


def _get_embedding_stats() -> dict:
    try:
        indexer = EmbeddingIndexer()
        if not indexer.exists():
            return {"status": "not_built"}

        indexer.load()
        return {
            "status": "ready",
            "shape": list(indexer.embeddings.shape),
            "total_docs": len(indexer.doc_ids),
            "memory_mb": round(indexer.embeddings.nbytes / (1024**2), 2)
        }
    except Exception as e:
        logger.error(f"Failed to get embedding stats: {e}")
        return {"status": "error", "message": str(e)}


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


def get_tfidf_status() -> dict:
    return _get_tfidf_stats()


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


def get_embedding_status() -> dict:
    return _get_embedding_stats()
