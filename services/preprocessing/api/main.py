from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from services.preprocessing.core.config import PreprocessingConfig
from services.preprocessing.core.pipeline import (
    preprocess as _core_preprocess,
    preprocess_batch as _core_preprocess_batch,
    get_preprocessor,
)
from services.preprocessing.core.resources import warmup_resources
from services.preprocessing.api.schemas import PreprocessRequest, PreprocessResponse, PreprocessBatchRequest, PreprocessBatchResponse
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


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    """
    Return request validation errors as a single readable string.

    FastAPI's default format puts a list of error objects in `detail`; here the
    messages are merged into one string so every 422 response in this service
    has a consistent, human-readable `detail`.
    """
    messages = "; ".join(
        str(err.get("msg", "")).removeprefix("Value error, ")
        for err in exc.errors()
    )
    return JSONResponse(status_code=422, content={"detail": messages})


@app.get("/health")
def health():
    return {"status": "ok", "service": "preprocessing"}


@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess_text(req: PreprocessRequest):
    config = None
    if req.config:
        try:
            config = PreprocessingConfig.from_dict(req.config)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    
    tokens = _core_preprocess(req.text, config)
    return PreprocessResponse(tokens=tokens, original=req.text)


@app.post("/preprocess/batch", response_model=PreprocessBatchResponse)
def preprocess_batch(req: PreprocessBatchRequest):
    """
    Batch preprocessing endpoint for efficient processing of multiple texts.
    Returns only tokens (no originals) for memory efficiency.

    Uses one spaCy streaming pass (nlp.pipe) over the whole batch and reuses
    a cached preprocessor for identical configs.
    """
    config = None
    if req.config:
        try:
            config = PreprocessingConfig.from_dict(req.config)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    results = _core_preprocess_batch(req.texts, config)
    return PreprocessBatchResponse(results=results, total=len(results))
