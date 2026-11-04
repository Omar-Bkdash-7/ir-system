"""
Index statistics service.

Provides functionality for retrieving statistics about built indices
including TF-IDF, Embedding, and BM25 indices.
"""

import logging
from typing import Dict, Any

from services.indexing.indexers import TFIDFIndexer, EmbeddingIndexer, BM25Indexer
from services.indexing.services.index_cache import load_cached


logger = logging.getLogger(__name__)


class TFIDFStatsRetriever:
    """Retrieves TF-IDF index statistics."""
    
    def get_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get TF-IDF index statistics.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
        
        Returns:
            Dictionary with index status and statistics
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        try:
            indexer = TFIDFIndexer(dataset_name=dataset_name)
            if not indexer.exists():
                return {"status": "not_built"}
            
            # The loaded index is cached (keyed by the files' modification time): status pages poll this endpoint
            indexer = load_cached(
                lambda: TFIDFIndexer(dataset_name=dataset_name).load(),
                [indexer.vectorizer_path, indexer.matrix_path, indexer.doc_ids_path],
            )
            return {
                "status": "ready",
                "vocab_size": len(indexer.vectorizer.vocabulary_),
                "matrix_shape": list(indexer.matrix.shape),
                "total_docs": len(indexer.doc_ids)
            }
        except Exception as e:
            logger.error(f"Failed to get TF-IDF stats: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


class EmbeddingStatsRetriever:
    """Retrieves embedding index statistics."""
    
    def get_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get Embedding index statistics.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
        
        Returns:
            Dictionary with index status and statistics
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        try:
            indexer = EmbeddingIndexer(dataset_name=dataset_name)
            if not indexer.exists():
                return {"status": "not_built"}
            
            indexer.load(mmap=True)  # only the shape and the size are needed: do not read the whole matrix
            return {
                "status": "ready",
                "shape": list(indexer.embeddings.shape),
                "total_docs": len(indexer.doc_ids),
                "memory_mb": round(indexer.embeddings.nbytes / (1024**2), 2)
            }
        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


class BM25StatsRetriever:
    """Retrieves BM25 index statistics."""
    
    def get_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get BM25 index statistics.
        
        Args:
            dataset_name: Dataset identifier (REQUIRED)
        
        Returns:
            Dictionary with index status and statistics
        """
        if not dataset_name:
            raise ValueError("dataset_name is required")
        
        try:
            indexer = BM25Indexer(dataset_name=dataset_name)
            if not indexer.exists():
                return {"status": "not_built"}
            
            indexer = load_cached(lambda: BM25Indexer(dataset_name=dataset_name).load(), [indexer.index_path])
            return {
                "status": "ready",
                "vocab_size": len(indexer.inverted_index),
                "total_docs": indexer.total_docs,
                "avg_doc_length": round(indexer.avg_doc_length, 2)
            }
        except Exception as e:
            logger.error(f"Failed to get BM25 stats: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


def get_index_stats(dataset_name: str, doc_count: int = None) -> Dict[str, Any]:
    """
    Public API: Get statistics for all indices.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        doc_count: Total document count (optional, for display purposes)
    
    Returns:
        Dictionary with dataset and index statistics
    """
    if not dataset_name:
        raise ValueError("dataset_name is required")
    
    tfidf_retriever = TFIDFStatsRetriever()
    embedding_retriever = EmbeddingStatsRetriever()
    bm25_retriever = BM25StatsRetriever()
    
    return {
        "dataset": {
            "total_docs": doc_count if doc_count is not None else "unknown",
            "name": dataset_name
        },
        "indices": {
            "tfidf": tfidf_retriever.get_stats(dataset_name),
            "embedding": embedding_retriever.get_stats(dataset_name),
            "bm25": bm25_retriever.get_stats(dataset_name)
        }
    }


def get_tfidf_stats(dataset_name: str) -> Dict[str, Any]:
    """
    Public API: Get TF-IDF index statistics.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
    
    Returns:
        Dictionary with TF-IDF index status and statistics
    """
    retriever = TFIDFStatsRetriever()
    return retriever.get_stats(dataset_name)


def get_embedding_stats(dataset_name: str) -> Dict[str, Any]:
    """
    Public API: Get embedding index statistics.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
    
    Returns:
        Dictionary with embedding index status and statistics
    """
    retriever = EmbeddingStatsRetriever()
    return retriever.get_stats(dataset_name)


def get_bm25_stats(dataset_name: str) -> Dict[str, Any]:
    """
    Public API: Get BM25 index statistics.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
    
    Returns:
        Dictionary with BM25 index status and statistics
    """
    retriever = BM25StatsRetriever()
    return retriever.get_stats(dataset_name)


__all__ = [
    "get_index_stats",
    "get_tfidf_stats",
    "get_embedding_stats",
    "get_bm25_stats",
    "TFIDFStatsRetriever",
    "EmbeddingStatsRetriever",
    "BM25StatsRetriever",
]
