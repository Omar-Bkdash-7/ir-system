"""
Indexing Service Package

Service-oriented architecture component for dataset indexing in the IR system.
Provides dataset validation, loading, and index building functionality.

Components:
    - dataset_service: Dataset management and PostgreSQL integration
    - indexing_service: Index building and statistics for TF-IDF, BM25, Embedding
    - indexers: Concrete indexer implementations
    - services: Supporting services for building and statistics

API Entry Points:
    - validate_and_store_metadata: Validate and store dataset metadata
    - upload_dataset_to_db: Upload documents to PostgreSQL
    - build_tfidf_index, build_embedding_index, build_bm25_index: Build indices
    - get_index_stats: Retrieve index statistics
"""

from services.indexing._lazy import install_lazy_exports

# Public names are resolved on first access (PEP 562), so importing this package no longer drags in psycopg2,
# ir_datasets, scikit-learn and numpy. Every old import path keeps working.
install_lazy_exports(globals(), __name__, {
    "validate_and_store_metadata": "services.indexing.dataset",
    "get_metadata": "services.indexing.dataset",
    "list_local_datasets": "services.indexing.dataset",
    "delete_local_dataset": "services.indexing.dataset",
    "upload_dataset_to_db": "services.indexing.dataset",
    "get_db_dataset_info": "services.indexing.dataset",
    "DatasetService": "services.indexing.dataset",
    "get_index_stats": "services.indexing.services",
    "get_tfidf_stats": "services.indexing.services",
    "get_embedding_stats": "services.indexing.services",
    "get_bm25_stats": "services.indexing.services",
    "build_tfidf_index": "services.indexing.services",
    "build_embedding_index": "services.indexing.services",
    "build_bm25_index": "services.indexing.services",
})

__all__ = [
    # Dataset management
    "validate_and_store_metadata",
    "get_metadata",
    "list_local_datasets",
    "delete_local_dataset",
    "upload_dataset_to_db",
    "get_db_dataset_info",
    "DatasetService",
    
    # Index management
    "get_index_stats",
    "get_tfidf_stats",
    "get_embedding_stats",
    "get_bm25_stats",
    "build_tfidf_index",
    "build_embedding_index",
    "build_bm25_index",
]

__version__ = "1.0.0"
__service_name__ = "indexing"
