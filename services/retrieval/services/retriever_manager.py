from services.retrieval.retrievers import TFIDFRetriever, EmbeddingRetriever, BM25Retriever
import logging

logger = logging.getLogger(__name__)

_tfidf_retrievers = {}
_embedding_retrievers = {}
_bm25_retrievers = {}


def get_tfidf_retriever(dataset: str = None) -> TFIDFRetriever:
    global _tfidf_retrievers
    
    if not dataset:
        raise ValueError("dataset parameter is required")
    
    if dataset not in _tfidf_retrievers:
        logger.info(f"Initializing TF-IDF retriever for dataset: {dataset}")
        _tfidf_retrievers[dataset] = TFIDFRetriever(dataset_name=dataset)
        _tfidf_retrievers[dataset].load()
        logger.info(f"TF-IDF retriever ready for dataset: {dataset}")
    
    return _tfidf_retrievers[dataset]


def get_embedding_retriever(dataset: str = None) -> EmbeddingRetriever:
    global _embedding_retrievers
    
    if not dataset:
        raise ValueError("dataset parameter is required")
    
    if dataset not in _embedding_retrievers:
        logger.info(f"Initializing Embedding retriever for dataset: {dataset}")
        _embedding_retrievers[dataset] = EmbeddingRetriever(dataset_name=dataset)
        _embedding_retrievers[dataset].load()
        logger.info(f"Embedding retriever ready for dataset: {dataset}")
    
    return _embedding_retrievers[dataset]


def get_bm25_retriever(dataset: str = None) -> BM25Retriever:
    global _bm25_retrievers
    
    if not dataset:
        raise ValueError("dataset parameter is required")
    
    if dataset not in _bm25_retrievers:
        logger.info(f"Initializing BM25 retriever for dataset: {dataset}")
        _bm25_retrievers[dataset] = BM25Retriever(dataset_name=dataset)
        _bm25_retrievers[dataset].load()
        logger.info(f"BM25 retriever ready for dataset: {dataset}")
    
    return _bm25_retrievers[dataset]
