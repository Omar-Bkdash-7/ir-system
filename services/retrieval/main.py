from fastapi import FastAPI
from services.retrieval.retrieval_service import retrieve_tfidf, get_tfidf_retriever
from services.retrieval.schemas import RetrievalRequest, RetrievalResponse
from common.exception_handlers import register_exception_handlers
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Retrieval Service")
register_exception_handlers(app)

@app.on_event("startup")
def startup_event():
    try:
        get_tfidf_retriever()
        logging.info("TF-IDF retriever loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load TF-IDF retriever: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "service": "retrieval"}

@app.post("/retrieve/tfidf", response_model=RetrievalResponse)
def retrieve_tfidf_endpoint(req: RetrievalRequest):
    results = retrieve_tfidf(req)
    return RetrievalResponse(results=results, total=len(results))

@app.get("/status/tfidf")
def tfidf_status():
    retriever = get_tfidf_retriever()
    return {
        "status": "ready",
        "total_docs": len(retriever.doc_ids),
        "vocab_size": len(retriever.vectorizer.vocabulary_)
    }
