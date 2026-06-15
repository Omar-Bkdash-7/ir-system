from services.indexing.tfidf_indexer import TFIDFIndexer
from services.indexing.dataset_service import get_metadata
import logging

logger = logging.getLogger(__name__)

def get_index_stats() -> dict:
    metadata = get_metadata()
    
    tfidf_indexer = TFIDFIndexer()
    tfidf_stats = None
    
    if tfidf_indexer.exists():
        tfidf_indexer.load()
        tfidf_stats = {
            "vocab_size": len(tfidf_indexer.vectorizer.vocabulary_),
            "matrix_shape": tfidf_indexer.matrix.shape,
            "total_docs": len(tfidf_indexer.doc_ids)
        }
    
    return {
        "total_docs": metadata["doc_count"],
        "avg_doc_length": None,
        "tfidf": tfidf_stats
    }

def build_tfidf_index(force_rebuild: bool = False) -> dict:
    indexer = TFIDFIndexer()
    
    if indexer.exists() and not force_rebuild:
        return {
            "status": "skipped",
            "message": "TF-IDF index already exists. Use force_rebuild=true to rebuild."
        }
    
    indexer.build()
    indexer.save()
    
    return {
        "status": "success",
        "vocab_size": len(indexer.vectorizer.vocabulary_),
        "matrix_shape": list(indexer.matrix.shape),
        "total_docs": len(indexer.doc_ids)
    }

def get_tfidf_status() -> dict:
    indexer = TFIDFIndexer()
    
    if not indexer.exists():
        return {
            "status": "not_built",
            "message": "TF-IDF index has not been built yet"
        }
    
    indexer.load()
    return {
        "status": "ready",
        "vocab_size": len(indexer.vectorizer.vocabulary_),
        "matrix_shape": list(indexer.matrix.shape),
        "total_docs": len(indexer.doc_ids)
    }
