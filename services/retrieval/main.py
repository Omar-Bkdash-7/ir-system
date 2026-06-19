from fastapi import FastAPI, HTTPException
from services.retrieval.retrieval_service import (
    retrieve,
    retrieve_tfidf,
    retrieve_embedding,
    retrieve_bm25,
    retrieve_hybrid_serial,
    get_tfidf_retriever,
    get_embedding_retriever,
    get_bm25_retriever
)
from services.retrieval.schemas import RetrievalRequest, RetrievalResponse, HybridSerialRequest
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
import logging

logger = setup_logging("retrieval")

app = FastAPI(title="Retrieval Service")
register_exception_handlers(app)


@app.on_event("startup")
def startup_event():
    logger.info("Loading indices on startup...")
    try:
        get_tfidf_retriever()
        logger.info("TF-IDF retriever loaded successfully")
    except Exception as e:
        logger.warning(f"TF-IDF retriever not available: {e}")
    
    try:
        get_embedding_retriever()
        logger.info("Embedding retriever loaded successfully")
    except Exception as e:
        logger.warning(f"Embedding retriever not available: {e}")
    
    try:
        get_bm25_retriever()
        logger.info("BM25 retriever loaded successfully")
    except Exception as e:
        logger.warning(f"BM25 retriever not available: {e}")
    
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
def tfidf_status():
    try:
        retriever = get_tfidf_retriever()
        return {
            "status": "ready",
            "total_docs": len(retriever.doc_ids),
            "vocab_size": len(retriever.vectorizer.vocabulary_),
            "loaded": retriever.loaded
        }
    except FileNotFoundError as e:
        logger.warning(f"TF-IDF index not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting TF-IDF status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/embedding")
def embedding_status():
    try:
        retriever = get_embedding_retriever()
        stats = retriever.get_statistics()
        return {
            "status": "ready",
            "loaded": retriever.loaded,
            **stats
        }
    except FileNotFoundError as e:
        logger.warning(f"Embedding index not found: {e}")
        return {
            "status": "not_built",
            "message": "Embedding index has not been built yet. Build it via /index/embedding/build endpoint on indexing service."
        }
    except Exception as e:
        logger.error(f"Error getting embedding status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def overall_status():
    tfidf_ok = False
    embedding_ok = False
    bm25_ok = False

    try:
        get_tfidf_retriever()
        tfidf_ok = True
    except:
        pass

    try:
        get_embedding_retriever()
        embedding_ok = True
    except:
        pass

    try:
        get_bm25_retriever()
        bm25_ok = True
    except:
        pass

    return {
        "service": "retrieval",
        "models": {
            "tfidf": "ready" if tfidf_ok else "unavailable",
            "embedding": "ready" if embedding_ok else "unavailable",
            "bm25": "ready" if bm25_ok else "unavailable"
        }
    }


@app.get("/status/bm25")
def bm25_status():
    try:
        retriever = get_bm25_retriever()
        stats = retriever.get_statistics()
        return {
            "status": "ready",
            "loaded": retriever.loaded,
            **stats
        }
    except FileNotFoundError as e:
        logger.warning(f"BM25 index not found: {e}")
        return {
            "status": "not_built",
            "message": "BM25 index has not been built yet. Build it via /index/bm25/build endpoint on indexing service."
        }
    except Exception as e:
        logger.error(f"Error getting BM25 status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve/hybrid/serial", response_model=RetrievalResponse)
def retrieve_hybrid_serial_endpoint(req: HybridSerialRequest):
    results = retrieve_hybrid_serial(req)
    return RetrievalResponse(
        results=results,
        total=len(results),
        model=f"hybrid_serial_{req.first_model}_{req.second_model}"
    )
