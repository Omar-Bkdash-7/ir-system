from contextlib import asynccontextmanager
from fastapi import FastAPI
from services.preprocessing.preprocessing_service import (
    preprocess,
    warmup_resources,
    PreprocessingConfig,
)
from services.preprocessing.api.schemas import PreprocessRequest, PreprocessResponse
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging

logger = setup_logging("preprocessing")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting preprocessing service...")
    warmup_resources(use_spacy=True, use_stopwords=True)
    logger.info("Preprocessing service ready")
    yield
    logger.info("Shutting down preprocessing service")


app = FastAPI(title="Preprocessing Service", lifespan=lifespan)
register_exception_handlers(app)


@app.get("/health")
def health():
    return {"status": "ok", "service": "preprocessing"}


@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess_text(req: PreprocessRequest):
    config = None
    if req.config:
        config = PreprocessingConfig(**req.config)
    
    tokens = preprocess(req.text, config)
    return PreprocessResponse(tokens=tokens, original=req.text)
