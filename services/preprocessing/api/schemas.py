from pydantic import BaseModel, Field
from typing import Optional, List


class PreprocessRequest(BaseModel):
    text: str
    config: Optional[dict] = Field(
        default=None,
        description="Preprocessing configuration options",
        examples=[{
            "normalize_unicode": True,
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": True,
            "min_token_length": 2,
            "max_token_length": None,
            "use_lemmatization": True,
            "use_stemming": False,
            "keep_numbers": False,
            "use_spacy_tokenizer": True,
            "custom_stopwords": []
        }]
    )


class PreprocessResponse(BaseModel):
    tokens: List[str]
    original: str
