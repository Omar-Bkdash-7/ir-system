from services.indexing.indexers import TFIDFIndexer, EmbeddingIndexer, BM25Indexer
from services.indexing.dataset_service import get_metadata
import logging

logger = logging.getLogger(__name__)


def get_index_stats() -> dict:
    try:
        metadata = get_metadata()
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        metadata = {"doc_count": "unknown"}

    tfidf_stats = get_tfidf_stats()
    embedding_stats = get_embedding_stats()
    bm25_stats = get_bm25_stats()

    return {
        "dataset": {
            "total_docs": metadata.get("doc_count", "unknown"),
            "name": metadata.get("dataset_name", "unknown")
        },
        "indices": {
            "tfidf": tfidf_stats,
            "embedding": embedding_stats,
            "bm25": bm25_stats
        }
    }


def get_tfidf_stats() -> dict:
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


def get_embedding_stats() -> dict:
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


def get_bm25_stats() -> dict:
    try:
        indexer = BM25Indexer()
        if not indexer.exists():
            return {"status": "not_built"}

        indexer.load()
        return {
            "status": "ready",
            "vocab_size": len(indexer.inverted_index),
            "total_docs": indexer.total_docs,
            "avg_doc_length": round(indexer.avg_doc_length, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get BM25 stats: {e}")
        return {"status": "error", "message": str(e)}
