from fastapi import FastAPI
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging

logger = setup_logging("query_refinement")

app = FastAPI(title="Query Refinement Service")
register_exception_handlers(app)

@app.get("/health")
def health():
    return {"status": "ok", "service": "query_refinement"}
