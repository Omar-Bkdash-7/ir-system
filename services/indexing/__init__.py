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

from services.indexing.dataset_service import (
    validate_and_store_metadata,
    get_metadata,
    list_local_datasets,
    delete_local_dataset,
    upload_dataset_to_db,
    get_db_dataset_info,
    DatasetService,
)

from services.indexing.indexing_service import (
    get_index_stats,
    get_tfidf_stats,
    get_embedding_stats,
    get_bm25_stats,
    build_tfidf_index,
    build_embedding_index,
    build_bm25_index,
)


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
