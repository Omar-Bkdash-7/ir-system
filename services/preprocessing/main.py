from fastapi import FastAPI
from services.preprocessing.preprocessing_service import preprocess
from services.preprocessing.schemas import PreprocessRequest, PreprocessResponse
from common.exception_handlers import register_exception_handlers
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Preprocessing Service")
register_exception_handlers(app)

@app.get("/health")
def health():
    return {"status": "ok", "service": "preprocessing"}

@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess_text(req: PreprocessRequest):
    tokens = preprocess(req.text, req.mode)
    return PreprocessResponse(tokens=tokens, original=req.text)
