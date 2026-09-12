from concurrent.futures import ThreadPoolExecutor, as_completed
from services.retrieval.schemas import RetrievalRequest, HybridParallelRequest
from services.retrieval.services.retriever_manager import (
    get_tfidf_retriever,
    get_embedding_retriever,
    get_bm25_retriever
)
from services.retrieval.services.basic_retrieval import retrieve, add_document_texts
from services.ranking.rankers import fuse_results
from common.exceptions import InvalidQueryError, RetrievalError
import logging

logger = logging.getLogger(__name__)


def retrieve_hybrid_parallel(request: HybridParallelRequest) -> list[dict]:
    dataset = request.dataset
    
    if not dataset:
        raise InvalidQueryError("dataset parameter is required")
    
    logger.info(f"Hybrid Parallel: models={request.models}, fusion={request.fusion_method}, "
                f"dataset={dataset}, candidates_per_model={request.candidates_per_model}, final_k={request.top_k}")
    
    if len(request.models) < 2:
        raise InvalidQueryError("Parallel mode requires at least 2 models")
    
    try:
        results_list = _retrieve_from_models_parallel(
            models=request.models,
            query_tokens=request.query_tokens,
            query_text=request.query_text,
            top_k=request.candidates_per_model,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
            dataset=dataset
        )
        
        logger.info(f"Retrieved from {len(results_list)} models, now fusing results")
        
        weights = None
        if request.fusion_method == "weighted" and request.model_weights:
            weights = [request.model_weights.get(model, 0.0) for model in request.models]
        
        fused_results = fuse_results(
            results_list=results_list,
            method=request.fusion_method,
            weights=weights,
            rrf_k=request.rrf_k
        )
        
        final_results = fused_results[:request.top_k]
        
        logger.info(f"Fusion completed: {len(fused_results)} unique docs, "
                   f"returning top {len(final_results)}")
        
        if request.include_text and final_results:
            add_document_texts(final_results, dataset)
        
        return final_results
    
    except InvalidQueryError:
        raise
    except Exception as e:
        logger.error(f"Hybrid parallel retrieval failed: {e}", exc_info=True)
        raise RetrievalError(f"Hybrid parallel retrieval failed: {str(e)}")


def _retrieve_from_models_parallel(
    models: list[str],
    query_tokens: list[str],
    query_text: str,
    top_k: int,
    bm25_k1: float,
    bm25_b: float,
    dataset: str
) -> list[list[dict]]:
    results_list = []
    
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {}
        
        for model in models:
            req = RetrievalRequest(
                query_tokens=query_tokens,
                query_text=query_text,
                model=model,
                top_k=top_k,
                include_text=False,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                dataset=dataset
            )
            future = executor.submit(_retrieve_single_model, req, model)
            future_to_model[future] = model
        
        model_results_map = {}
        
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                results = future.result()
                model_results_map[model] = results
                logger.debug(f"Model '{model}' retrieved {len(results)} results")
            except Exception as e:
                logger.error(f"Model '{model}' failed: {e}", exc_info=True)
                model_results_map[model] = []
    
    for model in models:
        results_list.append(model_results_map.get(model, []))
    
    return results_list


def _retrieve_single_model(req: RetrievalRequest, model: str) -> list[dict]:
    logger.debug(f"Starting retrieval for model: {model}")
    results = retrieve(req)
    logger.debug(f"Completed retrieval for model: {model}, got {len(results)} results")
    return results
