from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.rag.schemas import RAGRequest, RAGResponse
from services.rag.rag_service import RAGService
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
import os

logger = setup_logging("rag")

app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation service using HuggingFace Inference API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

rag_service = RAGService()


@app.get("/health")
def health():
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
    return {
        "status": "healthy",
        "service": "rag",
        "dependencies": {
            "retrieval": os.getenv("RETRIEVAL_URL", "http://retrieval:8003"),
            "huggingface_token": "configured" if hf_token else "not_configured",
            "remote_model": "flan-t5-large",
            "remote_available": bool(hf_token)
        }
    }


@app.post("/generate", response_model=RAGResponse)
async def generate_answer(request: RAGRequest):
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if not request.dataset or not request.dataset.strip():
            raise HTTPException(status_code=400, detail="Dataset must be specified")

        logger.info(
            f"Received RAG request: query='{request.query[:50]}...', "
            f"dataset={request.dataset}, model={request.generation_config.model}"
        )

        response = await rag_service.generate_answer(request)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing RAG request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models():
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
    return {
        "remote_model": {
            "id": "flan-t5-large",
            "name": "FLAN-T5 Large",
            "size": "780M parameters",
            "type": "remote",
            "quality": "Excellent",
            "speed": "Medium (5-10s)",
            "requirements": "Internet + HF Token",
            "available": bool(hf_token)
        },
        "retrieval_models": ["tfidf", "embedding", "bm25", "hybrid"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
