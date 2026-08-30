from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreprocessingConfig:
    normalize_unicode: bool = True
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 2
    max_token_length: Optional[int] = None
    use_lemmatization: bool = True
    use_stemming: bool = False
    keep_numbers: bool = False
    use_spacy_tokenizer: bool = False
    custom_stopwords: set[str] = field(default_factory=set)
    
    def to_dict(self) -> dict:
        return {
            "normalize_unicode": self.normalize_unicode,
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "remove_stopwords": self.remove_stopwords,
            "min_token_length": self.min_token_length,
            "max_token_length": self.max_token_length,
            "use_lemmatization": self.use_lemmatization,
            "use_stemming": self.use_stemming,
            "keep_numbers": self.keep_numbers,
            "use_spacy_tokenizer": self.use_spacy_tokenizer,
            "custom_stopwords": list(self.custom_stopwords),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessingConfig":
        custom_stopwords = set(data.pop("custom_stopwords", []))
        return cls(**data, custom_stopwords=custom_stopwords)
