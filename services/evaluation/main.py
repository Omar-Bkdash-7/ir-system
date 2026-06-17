from fastapi import FastAPI
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging

logger = setup_logging("evaluation")

app = FastAPI(title="Evaluation Service")
register_exception_handlers(app)

@app.get("/health")
def health():
    return {"status": "ok", "service": "evaluation"}
