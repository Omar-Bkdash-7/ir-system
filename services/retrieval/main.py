from fastapi import FastAPI, HTTPException
from services.retrieval.retrieval_service import (
    retrieve,
    retrieve_tfidf,
    retrieve_embedding,
    retrieve_bm25,
    retrieve_hybrid_serial,
    retrieve_hybrid_parallel,
    get_tfidf_retriever,
    get_embedding_retriever,
    get_bm25_retriever
)
from services.retrieval.schemas import (
    RetrievalRequest,
    RetrievalResponse,
    HybridSerialRequest,
    HybridParallelRequest
)
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
import logging

logger = setup_logging("retrieval")

app = FastAPI(title="Retrieval Service")
register_exception_handlers(app)


@app.on_event("startup")
def startup_event():
    logger.info("Retrieval service starting up...")
    logger.info("Retrievers will be loaded on-demand per dataset")
    logger.info("Startup complete")


@app.get("/health")
def health():
    return {"status": "ok", "service": "retrieval"}


@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve_endpoint(req: RetrievalRequest):
    results = retrieve(req)
    return RetrievalResponse(results=results, total=len(results), model=req.model)


@app.post("/retrieve/tfidf", response_model=RetrievalResponse)
def retrieve_tfidf_endpoint(req: RetrievalRequest):
    results = retrieve_tfidf(req)
    return RetrievalResponse(results=results, total=len(results), model="tfidf")


@app.post("/retrieve/embedding", response_model=RetrievalResponse)
def retrieve_embedding_endpoint(req: RetrievalRequest):
    results = retrieve_embedding(req)
    return RetrievalResponse(results=results, total=len(results), model="embedding")


@app.post("/retrieve/bm25", response_model=RetrievalResponse)
def retrieve_bm25_endpoint(req: RetrievalRequest):
    results = retrieve_bm25(req)
    return RetrievalResponse(results=results, total=len(results), model="bm25")


@app.get("/status/tfidf")
def tfidf_status(dataset: str = None):
    if not dataset:
        return {"error": "dataset parameter is required"}
    
    try:
        retriever = get_tfidf_retriever(dataset)
        return {
            "status": "ready",
            "dataset": dataset,
            "total_docs": len(retriever.doc_ids),
            "vocab_size": len(retriever.vectorizer.vocabulary_),
            "loaded": retriever.loaded
        }
    except FileNotFoundError as e:
        logger.warning(f"TF-IDF index not found for dataset {dataset}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting TF-IDF status for dataset {dataset}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/embedding")
def embedding_status(dataset: str = None):
    if not dataset:
        return {"error": "dataset parameter is required"}
    
    try:
        retriever = get_embedding_retriever(dataset)
        stats = retriever.get_statistics()
        return {
            "status": "ready",
            "dataset": dataset,
            "loaded": retriever.loaded,
            **stats
        }
    except FileNotFoundError as e:
        logger.warning(f"Embedding index not found for dataset {dataset}: {e}")
        return {
            "status": "not_built",
            "dataset": dataset,
            "message": "Embedding index has not been built yet. Build it via /index/embedding/build endpoint on indexing service."
        }
    except Exception as e:
        logger.error(f"Error getting embedding status for dataset {dataset}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def overall_status():
    return {
        "service": "retrieval",
        "message": "Retrievers loaded on-demand per dataset. Use /status/{model}?dataset=<name> for specific status."
    }


@app.get("/status/bm25")
def bm25_status(dataset: str = None):
    if not dataset:
        return {"error": "dataset parameter is required"}
    
    try:
        retriever = get_bm25_retriever(dataset)
        stats = retriever.get_statistics()
        return {
            "status": "ready",
            "dataset": dataset,
            "loaded": retriever.loaded,
            **stats
        }
    except FileNotFoundError as e:
        logger.warning(f"BM25 index not found for dataset {dataset}: {e}")
        return {
            "status": "not_built",
            "dataset": dataset,
            "message": "BM25 index has not been built yet. Build it via /index/bm25/build endpoint on indexing service."
        }
    except Exception as e:
        logger.error(f"Error getting BM25 status for dataset {dataset}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve/hybrid/serial", response_model=RetrievalResponse)
def retrieve_hybrid_serial_endpoint(req: HybridSerialRequest):
    results = retrieve_hybrid_serial(req)
    return RetrievalResponse(
        results=results,
        total=len(results),
        model=f"hybrid_serial_{req.first_model}_{req.second_model}"
    )


@app.post("/retrieve/hybrid/parallel", response_model=RetrievalResponse)
def retrieve_hybrid_parallel_endpoint(req: HybridParallelRequest):
    results = retrieve_hybrid_parallel(req)
    models_str = "_".join(req.models)
    return RetrievalResponse(
        results=results,
        total=len(results),
        model=f"hybrid_parallel_{models_str}_{req.fusion_method}"
    )
