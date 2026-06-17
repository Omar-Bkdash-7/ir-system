from fastapi import FastAPI
from services.api_gateway.dataset_client import get_dataset_info
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging

logger = setup_logging("api_gateway")

app = FastAPI(title="API Gateway")
register_exception_handlers(app)

@app.get("/health")
def health():
    return {"status": "ok", "service": "api_gateway"}

@app.get("/datasets/info")
def datasets_info():
    return get_dataset_info()
