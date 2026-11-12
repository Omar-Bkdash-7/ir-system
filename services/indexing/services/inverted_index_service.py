"""
Inverted Index Analysis Service.

Provides detailed insights into BM25 inverted index structure,
including term statistics, posting lists, and vocabulary analysis.
"""

import logging
from typing import Dict, Any, Optional

from services.indexing.indexers import BM25Indexer
from services.indexing.services.index_cache import load_cached


logger = logging.getLogger(__name__)


def _not_built() -> Dict[str, Any]:
    return {
        "status": "not_built",
        "message": "BM25 index has not been built yet"
    }


def _load_bm25(dataset_name: str) -> Optional[BM25Indexer]:
    """
    The loaded BM25 index, or None when it has not been built yet.

    The explorer calls these functions on every keystroke and each call used to unpickle the entire index from
    disk; the loaded index is now cached (keyed by the file's modification time, so a rebuild is noticed).
    """
    indexer = BM25Indexer(dataset_name=dataset_name)
    if not indexer.exists():
        return None
    return load_cached(lambda: BM25Indexer(dataset_name=dataset_name).load(), [indexer.index_path])


def get_inverted_index_stats(dataset_name: str) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the inverted index.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
    
    Returns:
        Dictionary containing:
        - vocab_size: Total number of unique terms
        - total_docs: Total number of documents
        - avg_doc_length: Average document length
        - total_postings: Total posting entries across all terms
        - avg_postings_per_term: Average posting list length
        - top_terms: Most frequent terms by document frequency
        - longest_posting_lists: Terms with most postings
    """
    if not dataset_name:
        raise ValueError("dataset_name is required")
    
    try:
        indexer = _load_bm25(dataset_name)
        
        if indexer is None:
            return _not_built()
        
        total_postings = sum(len(postings) for postings in indexer.inverted_index.values())
        avg_postings = total_postings / len(indexer.inverted_index) if indexer.inverted_index else 0
        
        top_terms_by_df = sorted(
            indexer.doc_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        longest_lists = sorted(
            [(term, len(indexer.inverted_index[term])) 
             for term in indexer.inverted_index.keys()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        doc_length_stats = _calculate_doc_length_stats(indexer)
        
        return {
            "status": "success",
            "vocab_size": len(indexer.inverted_index),
            "total_docs": indexer.total_docs,
            "avg_doc_length": round(indexer.avg_doc_length, 2),
            "total_postings": total_postings,
            "avg_postings_per_term": round(avg_postings, 2),
            "top_terms_by_df": [
                {"term": term, "doc_freq": df, "idf": round(indexer.idf_scores.get(term, 0), 4)}
                for term, df in top_terms_by_df
            ],
            "longest_posting_lists": [
                {"term": term, "posting_count": count}
                for term, count in longest_lists
            ],
            "doc_length_stats": doc_length_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get inverted index stats: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


def _calculate_doc_length_stats(indexer: BM25Indexer) -> Dict[str, Any]:
    """Calculate statistics about document lengths."""
    if not indexer.doc_lengths:
        return {}
    
    lengths = list(indexer.doc_lengths.values())
    lengths.sort()
    
    n = len(lengths)
    
    return {
        "min": lengths[0],
        "max": lengths[-1],
        "median": lengths[n // 2] if n > 0 else 0,
        "percentile_25": lengths[n // 4] if n > 0 else 0,
        "percentile_75": lengths[(3 * n) // 4] if n > 0 else 0,
    }


def get_term_info(dataset_name: str, term: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific term in the inverted index.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        term: The term to look up
        
    Returns:
        Dictionary containing:
        - term: The searched term
        - exists: Whether the term exists in vocabulary
        - doc_frequency: Number of documents containing the term
        - idf: IDF score
        - posting_list: List of (doc_id, term_freq) tuples
    """
    if not dataset_name:
        raise ValueError("dataset_name is required")
    
    try:
        indexer = _load_bm25(dataset_name)
        
        if indexer is None:
            return _not_built()
        
        if term not in indexer.inverted_index:
            return {
                "status": "success",
                "term": term,
                "exists": False,
                "message": f"Term '{term}' not found in vocabulary"
            }
        
        postings = indexer.get_posting_list(term)
        doc_freq = indexer.doc_frequencies.get(term, 0)
        idf = indexer.idf_scores.get(term, 0.0)
        
        posting_list_sample = [
            {"doc_id": entry.doc_id, "term_freq": entry.term_freq}
            for entry in postings[:100]
        ]
        
        return {
            "status": "success",
            "term": term,
            "exists": True,
            "doc_frequency": doc_freq,
            "idf": round(idf, 4),
            "total_postings": len(postings),
            "posting_list_sample": posting_list_sample,
            "sample_size": len(posting_list_sample),
            "total_size": len(postings)
        }
        
    except Exception as e:
        logger.error(f"Failed to get term info for '{term}': {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


def search_terms(dataset_name: str, prefix: str, limit: int = 50) -> Dict[str, Any]:
    """
    Search for terms in vocabulary that start with the given prefix.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        prefix: Prefix to search for
        limit: Maximum number of results
        
    Returns:
        Dictionary containing matching terms and their statistics
    """
    if not dataset_name:
        raise ValueError("dataset_name is required")
    
    try:
        indexer = _load_bm25(dataset_name)
        
        if indexer is None:
            return _not_built()
        
        matching_terms = [
            term for term in indexer.inverted_index.keys()
            if term.startswith(prefix.lower())
        ]
        
        matching_terms.sort()
        limited_terms = matching_terms[:limit]
        
        results = [
            {
                "term": term,
                "doc_freq": indexer.doc_frequencies.get(term, 0),
                "idf": round(indexer.idf_scores.get(term, 0), 4),
                "posting_count": len(indexer.inverted_index[term])
            }
            for term in limited_terms
        ]
        
        return {
            "status": "success",
            "prefix": prefix,
            "total_matches": len(matching_terms),
            "returned": len(results),
            "limit": limit,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to search terms with prefix '{prefix}': {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


def get_vocabulary_sample(dataset_name: str, sample_size: int = 100) -> Dict[str, Any]:
    """
    Get a random sample of vocabulary terms.
    
    Args:
        dataset_name: Dataset identifier (REQUIRED)
        sample_size: Number of terms to sample
        
    Returns:
        Dictionary containing sampled terms and statistics
    """
    if not dataset_name:
        raise ValueError("dataset_name is required")
    
    try:
        indexer = _load_bm25(dataset_name)
        
        if indexer is None:
            return _not_built()
        
        import random
        
        all_terms = list(indexer.inverted_index.keys())
        sample = random.sample(all_terms, min(sample_size, len(all_terms)))
        sample.sort()
        
        results = [
            {
                "term": term,
                "doc_freq": indexer.doc_frequencies.get(term, 0),
                "idf": round(indexer.idf_scores.get(term, 0), 4)
            }
            for term in sample
        ]
        
        return {
            "status": "success",
            "vocab_size": len(all_terms),
            "sample_size": len(results),
            "sample": results
        }
        
    except Exception as e:
        logger.error(f"Failed to get vocabulary sample: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


__all__ = [
    "get_inverted_index_stats",
    "get_term_info",
    "search_terms",
    "get_vocabulary_sample"
]
