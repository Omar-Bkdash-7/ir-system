import logging
from typing import Optional
from ..core.resources import get_spacy_model
from ..core.exceptions import TokenizationError

logger = logging.getLogger(__name__)


class SpacyTokenizer:
    def __init__(self, use_lemmatization: bool = True):
        self.use_lemmatization = use_lemmatization
        self.nlp = get_spacy_model()
    
    def tokenize(self, text: str) -> list[str]:
        try:
            doc = self.nlp(text)
            if self.use_lemmatization:
                return [token.lemma_ for token in doc]
            else:
                return [token.text for token in doc]
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise TokenizationError(f"Failed to tokenize text: {e}") from e


class SimpleTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()
