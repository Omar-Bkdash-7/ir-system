"""
Indexing Services Subpackage

Contains supporting services for the indexing service:
    - build_service: Index building operations
    - stats_service: Index statistics and monitoring
"""

from services.indexing._lazy import install_lazy_exports

# Public names are resolved on first access (PEP 562), so importing this package no longer drags in psycopg2,
# ir_datasets, scikit-learn and numpy. Every old import path keeps working.
install_lazy_exports(globals(), __name__, {
    "build_tfidf_index": "services.indexing.services.build_service",
    "build_embedding_index": "services.indexing.services.build_service",
    "build_bm25_index": "services.indexing.services.build_service",
    "build_all_indices": "services.indexing.services.build_service",
    "TFIDFIndexBuilder": "services.indexing.services.build_service",
    "EmbeddingIndexBuilder": "services.indexing.services.build_service",
    "BM25IndexBuilder": "services.indexing.services.build_service",
    "get_index_stats": "services.indexing.services.stats_service",
    "get_tfidf_stats": "services.indexing.services.stats_service",
    "get_embedding_stats": "services.indexing.services.stats_service",
    "get_bm25_stats": "services.indexing.services.stats_service",
    "TFIDFStatsRetriever": "services.indexing.services.stats_service",
    "EmbeddingStatsRetriever": "services.indexing.services.stats_service",
    "BM25StatsRetriever": "services.indexing.services.stats_service",
})

__all__ = [
    # Build services
    "build_tfidf_index",
    "build_embedding_index",
    "build_bm25_index",
    "build_all_indices",
    "TFIDFIndexBuilder",
    "EmbeddingIndexBuilder",
    "BM25IndexBuilder",
    
    # Stats services
    "get_index_stats",
    "get_tfidf_stats",
    "get_embedding_stats",
    "get_bm25_stats",
    "TFIDFStatsRetriever",
    "EmbeddingStatsRetriever",
    "BM25StatsRetriever",
]
