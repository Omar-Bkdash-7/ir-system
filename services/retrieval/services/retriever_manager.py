from services.retrieval.retrievers import TFIDFRetriever, EmbeddingRetriever, BM25Retriever
import logging

logger = logging.getLogger(__name__)

_tfidf_retriever = None
_embedding_retriever = None
_bm25_retriever = None


def get_tfidf_retriever() -> TFIDFRetriever:
    global _tfidf_retriever
    if _tfidf_retriever is None:
        logger.info("Initializing TF-IDF retriever")
        _tfidf_retriever = TFIDFRetriever()
        _tfidf_retriever.load()
        logger.info("TF-IDF retriever ready")
    return _tfidf_retriever


def get_embedding_retriever() -> EmbeddingRetriever:
    global _embedding_retriever
    if _embedding_retriever is None:
        logger.info("Initializing Embedding retriever")
        _embedding_retriever = EmbeddingRetriever()
        _embedding_retriever.load()
        logger.info("Embedding retriever ready")
    return _embedding_retriever


def get_bm25_retriever() -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is None:
        logger.info("Initializing BM25 retriever")
        _bm25_retriever = BM25Retriever()
        _bm25_retriever.load()
        logger.info("BM25 retriever ready")
    return _bm25_retriever
