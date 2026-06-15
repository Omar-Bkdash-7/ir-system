from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from services.preprocessing.preprocessing_service import preprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Preprocessing Service")

class PreprocessRequest(BaseModel):
    text: str
    mode: Literal["document", "query"] = "document"

class PreprocessResponse(BaseModel):
    tokens: list[str]
    original: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "preprocessing"}

@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess_text(req: PreprocessRequest):
    tokens = preprocess(req.text, req.mode)
    return PreprocessResponse(tokens=tokens, original=req.text)
