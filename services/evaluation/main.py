from fastapi import FastAPI
from common.exception_handlers import register_exception_handlers
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Evaluation Service")
register_exception_handlers(app)

@app.get("/health")
def health():
    return {"status": "ok", "service": "evaluation"}
