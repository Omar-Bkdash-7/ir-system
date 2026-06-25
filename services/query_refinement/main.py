from fastapi import FastAPI, HTTPException
from services.query_refinement.schemas import RefineRequest, RefineResponse
from services.query_refinement.services import refine_query, get_service
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging

logger = setup_logging("query_refinement")

app = FastAPI(
    title="Query Refinement Service",
    description="Intelligent query refinement with spelling correction, synonym expansion, and history-aware boosting"
)
register_exception_handlers(app)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Query Refinement Service...")
    try:
        get_service()
        logger.info("Query Refinement Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}", exc_info=True)

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "service": "query_refinement",
        "features": [
            "spelling_correction",
            "synonym_expansion",
            "history_boosting",
            "semantic_similarity"
        ]
    }

@app.post("/refine", response_model=RefineResponse)
def refine_endpoint(req: RefineRequest):
    try:
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        result = refine_query(req.query, req.history, req.enabled, req.session_id)
        return RefineResponse(**result)
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query refinement failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
