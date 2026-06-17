from services.retrieval.tfidf_retriever import TFIDFRetriever
from services.retrieval.schemas import RetrievalRequest
from common.database import get_documents_by_ids
import logging

logger = logging.getLogger(__name__)

_tfidf_retriever = None

def get_tfidf_retriever() -> TFIDFRetriever:
    global _tfidf_retriever
    if _tfidf_retriever is None:
        _tfidf_retriever = TFIDFRetriever()
        _tfidf_retriever.load()
    return _tfidf_retriever

def retrieve_tfidf(request: RetrievalRequest) -> list[dict]:
    retriever = get_tfidf_retriever()
    
    candidates_k = request.top_k * 20
    candidates = retriever.retrieve(request.query_tokens, candidates_k)
    
    top_results = candidates[:request.top_k]
    
    if request.include_text and top_results:
        doc_ids = [r["doc_id"] for r in top_results]
        texts = get_documents_by_ids(doc_ids)
        
        for result in top_results:
            result["text"] = texts.get(result["doc_id"], "")
    
    return top_results
