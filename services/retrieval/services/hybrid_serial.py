from services.retrieval.schemas import RetrievalRequest, HybridSerialRequest
from services.retrieval.services.retriever_manager import (
    get_tfidf_retriever,
    get_embedding_retriever,
    get_bm25_retriever
)
from services.retrieval.services.basic_retrieval import retrieve, add_document_texts
from common.exceptions import InvalidQueryError, RetrievalError
import logging

logger = logging.getLogger(__name__)


def retrieve_hybrid_serial(request: HybridSerialRequest) -> list[dict]:
    logger.info(f"Hybrid Serial: {request.first_model} -> {request.second_model}, "
                f"first_k={request.first_stage_k}, final_k={request.final_top_k}")
    
    if request.first_model == request.second_model:
        logger.warning(f"Same model for both stages: {request.first_model}. This is not recommended.")
    
    try:
        stage1_request = RetrievalRequest(
            query_tokens=request.query_tokens,
            query_text=request.query_text,
            model=request.first_model,
            top_k=request.first_stage_k,
            include_text=False,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b
        )
        
        logger.info(f"Stage 1: Retrieving {request.first_stage_k} candidates using {request.first_model}")
        stage1_results = retrieve(stage1_request)
        
        if not stage1_results:
            logger.warning("Stage 1 returned 0 results, skipping stage 2")
            return []
        
        logger.info(f"Stage 1 completed: {len(stage1_results)} candidates retrieved")
        
        candidate_doc_ids = [r["doc_id"] for r in stage1_results]
        
        logger.info(f"Stage 2: Re-ranking {len(candidate_doc_ids)} candidates using {request.second_model}")
        stage2_results = _rerank_candidates(
            candidate_doc_ids=candidate_doc_ids,
            query_tokens=request.query_tokens,
            query_text=request.query_text,
            model=request.second_model,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b
        )
        
        final_results = stage2_results[:request.final_top_k]
        
        logger.info(f"Stage 2 completed: returning top {len(final_results)} results")
        
        if request.include_text and final_results:
            add_document_texts(final_results)
        
        return final_results
    
    except InvalidQueryError:
        raise
    except Exception as e:
        logger.error(f"Hybrid serial retrieval failed: {e}", exc_info=True)
        raise RetrievalError(f"Hybrid serial retrieval failed: {str(e)}")


def _rerank_candidates(
    candidate_doc_ids: list[str],
    query_tokens: list[str],
    query_text: str,
    model: str,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75
) -> list[dict]:
    if model == "tfidf":
        retriever = get_tfidf_retriever()
        scores = retriever.score_documents(query_tokens, candidate_doc_ids)
    elif model == "embedding":
        retriever = get_embedding_retriever()
        scores = retriever.score_documents(query_text, candidate_doc_ids)
    elif model == "bm25":
        retriever = get_bm25_retriever()
        scores = retriever.score_documents(query_tokens, candidate_doc_ids, k1=bm25_k1, b=bm25_b)
    else:
        raise InvalidQueryError(f"Unknown model for re-ranking: {model}")
    
    results = [{"doc_id": doc_id, "score": score} for doc_id, score in scores.items()]
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results
