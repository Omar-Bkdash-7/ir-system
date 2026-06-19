from services.retrieval.tfidf_retriever import TFIDFRetriever
from services.retrieval.embedding_retriever import EmbeddingRetriever
from services.retrieval.bm25_retriever import BM25Retriever
from services.retrieval.schemas import RetrievalRequest
from common.database import get_documents_by_ids
from common.exceptions import InvalidQueryError, RetrievalError
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


def retrieve(request: RetrievalRequest) -> list[dict]:
    model = request.model.lower()

    logger.info(f"Retrieving with model: {model}, top_k: {request.top_k}")

    try:
        if model == "tfidf":
            results = _retrieve_tfidf(request)
        elif model == "embedding":
            results = _retrieve_embedding(request)
        elif model == "bm25":
            results = _retrieve_bm25(request)
        else:
            raise InvalidQueryError(f"Unknown retrieval model: {model}")

        if request.include_text and results:
            _add_document_texts(results)

        logger.info(f"Retrieved {len(results)} documents successfully")
        return results

    except InvalidQueryError:
        raise
    except Exception as e:
        logger.error(f"Retrieval failed with model {model}: {e}", exc_info=True)
        raise RetrievalError(f"Retrieval failed: {str(e)}")


def _retrieve_tfidf(request: RetrievalRequest) -> list[dict]:
    if not request.query_tokens:
        raise InvalidQueryError("query_tokens is required for TF-IDF retrieval")

    retriever = get_tfidf_retriever()
    candidates_k = min(request.top_k * 20, len(retriever.doc_ids))

    logger.debug(f"TF-IDF: retrieving {candidates_k} candidates for {len(request.query_tokens)} tokens")

    candidates = retriever.retrieve(request.query_tokens, candidates_k)
    results = candidates[:request.top_k]
    
    logger.debug(f"TF-IDF: narrowed down to top {len(results)} results")
    return results


def _retrieve_embedding(request: RetrievalRequest) -> list[dict]:
    if not request.query_text or not request.query_text.strip():
        raise InvalidQueryError("query_text is required for embedding retrieval")

    retriever = get_embedding_retriever()
    candidates_k = min(request.top_k * 20, len(retriever.doc_ids))

    logger.debug(f"Embedding: retrieving {candidates_k} candidates")

    candidates = retriever.retrieve(request.query_text, candidates_k)
    results = candidates[:request.top_k]

    logger.debug(f"Embedding: narrowed down to top {len(results)} results")
    return results


def _retrieve_bm25(request: RetrievalRequest) -> list[dict]:
    if not request.query_tokens:
        raise InvalidQueryError("query_tokens is required for BM25 retrieval")

    retriever = get_bm25_retriever()
    candidates_k = min(request.top_k * 20, len(retriever.doc_ids))
    
    k1 = getattr(request, 'bm25_k1', 1.5)
    b = getattr(request, 'bm25_b', 0.75)

    logger.debug(f"BM25: retrieving {candidates_k} candidates with k1={k1}, b={b}")

    candidates = retriever.retrieve(request.query_tokens, candidates_k, k1=k1, b=b)
    results = candidates[:request.top_k]

    logger.debug(f"BM25: narrowed down to top {len(results)} results")
    return results


def _add_document_texts(results: list[dict]) -> None:
    doc_ids = [r["doc_id"] for r in results]

    logger.debug(f"Fetching texts for {len(doc_ids)} documents from database")

    try:
        texts = get_documents_by_ids(doc_ids)

        for result in results:
            result["text"] = texts.get(result["doc_id"], "")

        missing = sum(1 for r in results if not r["text"])
        if missing > 0:
            logger.warning(f"{missing} documents have missing texts")

    except Exception as e:
        logger.error(f"Failed to fetch document texts: {e}", exc_info=True)
        for result in results:
            result["text"] = ""


def retrieve_tfidf(request: RetrievalRequest) -> list[dict]:
    request.model = "tfidf"
    return retrieve(request)


def retrieve_embedding(request: RetrievalRequest) -> list[dict]:
    request.model = "embedding"
    return retrieve(request)


def retrieve_bm25(request: RetrievalRequest) -> list[dict]:
    request.model = "bm25"
    return retrieve(request)
