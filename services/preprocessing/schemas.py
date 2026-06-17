from pydantic import BaseModel
from typing import Literal, List

class PreprocessRequest(BaseModel):
    text: str
    mode: Literal["document", "query"] = "document"

class PreprocessResponse(BaseModel):
    tokens: List[str]
    original: str
