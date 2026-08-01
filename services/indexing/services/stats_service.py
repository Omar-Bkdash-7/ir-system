"""
Index statistics service.

Provides functionality for retrieving statistics about built indices
including TF-IDF, Embedding, and BM25 indices.
"""

import logging
from typing import Dict, Any

from services.indexing.indexers import TFIDFIndexer, EmbeddingIndexer, BM25Indexer
from services.indexing.dataset_service import get_metadata


logger = logging.getLogger(__name__)


class IndexStatsRetriever:
    """Base class for retrieving index statistics."""
    
    @staticmethod
    def _safe_get_metadata() -> Dict[str, Any]:
        """
        Safely retrieve metadata with fallback on error.
        
        Returns:
            Metadata dictionary or fallback with unknown values
        """
        try:
            return get_metadata()
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {"doc_count": "unknown", "dataset_name": "unknown"}


class TFIDFStatsRetriever(IndexStatsRetriever):
    """Retrieves TF-IDF index statistics."""
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get TF-IDF index statistics.
        
        Returns:
            Dictionary with index status and statistics
        """
        try:
            # Get dataset name from metadata
            metadata = self._safe_get_metadata()
            dataset_name = metadata.get("dataset_name")
            
            indexer = TFIDFIndexer(dataset_name=dataset_name)
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
            logger.error(f"Failed to get TF-IDF stats: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


class EmbeddingStatsRetriever(IndexStatsRetriever):
    """Retrieves embedding index statistics."""
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get Embedding index statistics.
        
        Returns:
            Dictionary with index status and statistics
        """
        try:
            # Get dataset name from metadata
            metadata = self._safe_get_metadata()
            dataset_name = metadata.get("dataset_name")
            
            indexer = EmbeddingIndexer(dataset_name=dataset_name)
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
            logger.error(f"Failed to get embedding stats: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


class BM25StatsRetriever(IndexStatsRetriever):
    """Retrieves BM25 index statistics."""
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get BM25 index statistics.
        
        Returns:
            Dictionary with index status and statistics
        """
        try:
            # Get dataset name from metadata
            metadata = self._safe_get_metadata()
            dataset_name = metadata.get("dataset_name")
            
            indexer = BM25Indexer(dataset_name=dataset_name)
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
            logger.error(f"Failed to get BM25 stats: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


class AllIndicesStatsRetriever(IndexStatsRetriever):
    """Retrieves statistics for all indices."""
    
    def __init__(self):
        self.tfidf_retriever = TFIDFStatsRetriever()
        self.embedding_retriever = EmbeddingStatsRetriever()
        self.bm25_retriever = BM25StatsRetriever()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all indices.
        
        Returns:
            Dictionary with dataset info and all index statistics
        """
        metadata = self._safe_get_metadata()
        
        return {
            "dataset": {
                "total_docs": metadata.get("doc_count", "unknown"),
                "name": metadata.get("dataset_name", "unknown")
            },
            "indices": {
                "tfidf": self.tfidf_retriever.get_stats(),
                "embedding": self.embedding_retriever.get_stats(),
                "bm25": self.bm25_retriever.get_stats()
            }
        }


def get_index_stats() -> Dict[str, Any]:
    """
    Public API: Get statistics for all indices.
    
    Returns:
        Dictionary with dataset and index statistics
    """
    retriever = AllIndicesStatsRetriever()
    return retriever.get_stats()


def get_tfidf_stats() -> Dict[str, Any]:
    """
    Public API: Get TF-IDF index statistics.
    
    Returns:
        Dictionary with TF-IDF index status and statistics
    """
    retriever = TFIDFStatsRetriever()
    return retriever.get_stats()


def get_embedding_stats() -> Dict[str, Any]:
    """
    Public API: Get embedding index statistics.
    
    Returns:
        Dictionary with embedding index status and statistics
    """
    retriever = EmbeddingStatsRetriever()
    return retriever.get_stats()


def get_bm25_stats() -> Dict[str, Any]:
    """
    Public API: Get BM25 index statistics.
    
    Returns:
        Dictionary with BM25 index status and statistics
    """
    retriever = BM25StatsRetriever()
    return retriever.get_stats()


__all__ = [
    "get_index_stats",
    "get_tfidf_stats",
    "get_embedding_stats",
    "get_bm25_stats",
    "TFIDFStatsRetriever",
    "EmbeddingStatsRetriever",
    "BM25StatsRetriever",
    "AllIndicesStatsRetriever",
]
