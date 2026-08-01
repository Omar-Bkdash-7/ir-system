"""
Indexing service package.

Provides dataset management, indexing, and building functionality
for information retrieval system.
"""

from services.indexing.services import (
    get_index_stats,
    get_tfidf_stats,
    get_embedding_stats,
    get_bm25_stats,
    build_tfidf_index,
    build_embedding_index,
    build_bm25_index
)

get_tfidf_status = get_tfidf_stats
get_embedding_status = get_embedding_stats
get_bm25_status = get_bm25_stats

__all__ = [
    "get_index_stats",
    "get_tfidf_stats",
    "get_embedding_stats",
    "get_bm25_stats",
    "get_tfidf_status",
    "get_embedding_status",
    "get_bm25_status",
    "build_tfidf_index",
    "build_embedding_index",
    "build_bm25_index"
]
