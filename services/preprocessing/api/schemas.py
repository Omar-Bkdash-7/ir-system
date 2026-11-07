from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class PreprocessRequest(BaseModel):
    text: str = Field(..., max_length=100000, description="Text to preprocess (max 100K chars)")
    mode: Optional[str] = Field(
        default=None,
        description="Processing mode: 'query' or 'document' (for future use)"
    )
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


class PreprocessBatchRequest(BaseModel):
    texts: List[str] = Field(
        ..., 
        max_length=1000,
        description="List of texts to preprocess (max 1000 texts)"
    )
    mode: Optional[str] = Field(
        default=None,
        description="Processing mode: 'query' or 'document' (for future use)"
    )
    config: Optional[dict] = Field(
        default=None,
        description="Preprocessing configuration options (shared for all texts)"
    )
    
    @field_validator('texts')
    @classmethod
    def validate_text_lengths(cls, v):
        for i, text in enumerate(v):
            if len(text) > 100000:
                raise ValueError(f"Text at index {i} exceeds maximum length of 100K characters")
        return v


class PreprocessResponse(BaseModel):
    tokens: List[str]
    original: Optional[str] = Field(
        default=None,
        description="Original text (omitted in batch responses for efficiency)"
    )


class PreprocessBatchResponse(BaseModel):
    results: List[List[str]] = Field(
        description="List of token lists (originals omitted for efficiency)"
    )
    total: int = Field(description="Total number of texts processed")
