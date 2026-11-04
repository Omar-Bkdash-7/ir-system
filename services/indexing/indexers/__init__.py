"""
Indexers package.

Provides indexer implementations for different retrieval algorithms:
- TF-IDF: Term Frequency-Inverse Document Frequency
- Embedding: Dense vector embeddings using transformer models
- BM25: Best Match 25 probabilistic ranking
"""

from services.indexing._lazy import install_lazy_exports

# Public names are resolved on first access (PEP 562), so importing this package no longer drags in psycopg2,
# ir_datasets, scikit-learn and numpy. Every old import path keeps working.
install_lazy_exports(globals(), __name__, {
    "TFIDFIndexer": "services.indexing.indexers.tfidf_indexer",
    "EmbeddingIndexer": "services.indexing.indexers.embedding_indexer",
    "BM25Indexer": "services.indexing.indexers.bm25_indexer",
})

__all__ = [
    "TFIDFIndexer",
    "EmbeddingIndexer",
    "BM25Indexer",
]
