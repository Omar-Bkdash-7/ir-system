"""Index statistics and inverted-index analysis.

These are read paths: a broken index is reported inside the answer as
`{"status": "error", ...}` (the UI renders it), never as a 500. The loaded index is
cached by file modification time, so polling endpoints do not unpickle it again.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StatsService:
    """Statistics of the three indices plus the BM25 vocabulary explorer."""

    # ----------------------------------------------------------------------------------------------- indices
    def all_indices(self, dataset_name: str, doc_count: Optional[int] = None) -> Dict[str, Any]:
        from services.indexing.services.stats_service import get_index_stats

        return get_index_stats(dataset_name, doc_count)

    def tfidf(self, dataset_name: str) -> Dict[str, Any]:
        from services.indexing.services.stats_service import get_tfidf_stats

        return get_tfidf_stats(dataset_name)

    def embedding(self, dataset_name: str) -> Dict[str, Any]:
        from services.indexing.services.stats_service import get_embedding_stats

        return get_embedding_stats(dataset_name)

    def bm25(self, dataset_name: str) -> Dict[str, Any]:
        from services.indexing.services.stats_service import get_bm25_stats

        return get_bm25_stats(dataset_name)

    # ----------------------------------------------------------------------------------------------- vocabulary
    def inverted_index(self, dataset_name: str) -> Dict[str, Any]:
        from services.indexing.services.inverted_index_service import get_inverted_index_stats

        return get_inverted_index_stats(dataset_name)

    def term(self, dataset_name: str, term: str) -> Dict[str, Any]:
        from services.indexing.services.inverted_index_service import get_term_info

        return get_term_info(dataset_name, term)

    def search_terms(self, dataset_name: str, prefix: str, limit: int = 50) -> Dict[str, Any]:
        from services.indexing.services.inverted_index_service import search_terms

        return search_terms(dataset_name, prefix, limit)

    def vocabulary_sample(self, dataset_name: str, sample_size: int = 100) -> Dict[str, Any]:
        from services.indexing.services.inverted_index_service import get_vocabulary_sample

        return get_vocabulary_sample(dataset_name, sample_size)
