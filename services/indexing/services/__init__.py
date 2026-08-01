"""
Indexing Services Subpackage

Contains supporting services for the indexing service:
    - build_service: Index building operations
    - stats_service: Index statistics and monitoring
"""

from services.indexing.services.build_service import (
    build_tfidf_index,
    build_embedding_index,
    build_bm25_index,
    TFIDFIndexBuilder,
    EmbeddingIndexBuilder,
    BM25IndexBuilder,
)

from services.indexing.services.stats_service import (
    get_index_stats,
    get_tfidf_stats,
    get_embedding_stats,
    get_bm25_stats,
    TFIDFStatsRetriever,
    EmbeddingStatsRetriever,
    BM25StatsRetriever,
    AllIndicesStatsRetriever,
)


__all__ = [
    # Build services
    "build_tfidf_index",
    "build_embedding_index",
    "build_bm25_index",
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
    "AllIndicesStatsRetriever",
]
