import logging
from typing import Optional
from .config import PreprocessingConfig
from ..processors.normalizer import TextNormalizer
from ..processors.tokenizer import SpacyTokenizer, SimpleTokenizer
from ..processors.filters import TokenFilter
from ..processors.lemmatizers import WordNetLemmatizerWrapper

logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        
        self.normalizer = TextNormalizer(
            normalize_unicode=self.config.normalize_unicode,
            lowercase=self.config.lowercase,
        )
        
        if self.config.use_spacy_tokenizer:
            self.tokenizer = SpacyTokenizer(use_lemmatization=self.config.use_lemmatization)
        else:
            self.tokenizer = SimpleTokenizer()
        
        self.filter = TokenFilter(
            remove_punctuation=self.config.remove_punctuation,
            remove_stopwords=self.config.remove_stopwords,
            keep_numbers=self.config.keep_numbers,
            min_token_length=self.config.min_token_length,
            max_token_length=self.config.max_token_length,
            custom_stopwords=self.config.custom_stopwords,
        )
        
        if not self.config.use_spacy_tokenizer and self.config.use_lemmatization:
            self.lemmatizer = WordNetLemmatizerWrapper()
        else:
            self.lemmatizer = None
    
    def preprocess(self, text: str) -> list[str]:
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text received")
            return []
        
        text = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(text)
        tokens = self.filter.filter_tokens(tokens)
        
        if self.lemmatizer:
            tokens = self.lemmatizer.lemmatize(tokens)
        
        return tokens


_default_preprocessor: Optional[TextPreprocessor] = None


def get_default_preprocessor() -> TextPreprocessor:
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = TextPreprocessor()
    return _default_preprocessor


def preprocess(text: str, config: Optional[PreprocessingConfig] = None) -> list[str]:
    if config is None:
        preprocessor = get_default_preprocessor()
    else:
        preprocessor = TextPreprocessor(config)
    
    return preprocessor.preprocess(text)
