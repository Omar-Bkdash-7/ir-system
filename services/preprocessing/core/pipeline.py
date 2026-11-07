import logging
from collections import OrderedDict
from threading import Lock
from typing import Optional

from .config import PreprocessingConfig
from ..processors.normalizer import TextNormalizer
from ..processors.tokenizer import SpacyTokenizer, SimpleTokenizer
from ..processors.filters import TokenFilter
from ..processors.lemmatizers import WordNetLemmatizerWrapper
from ..processors.stemmers import PorterStemmerWrapper

logger = logging.getLogger(__name__)

# Number of distinct config-keyed preprocessors kept cached for reuse.
_PREPROCESSOR_CACHE_SIZE = 8


class TextPreprocessor:
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

        self.normalizer = TextNormalizer(
            normalize_unicode=self.config.normalize_unicode,
            lowercase=self.config.lowercase,
        )

        # Determine morphological processing strategy
        # Stemming and Lemmatization are mutually exclusive
        use_spacy_lemma = False
        use_wordnet_lemma = False
        use_porter_stem = False

        if self.config.use_stemming:
            # Stemming: use Porter stemmer after tokenization, no lemmatization at all
            use_porter_stem = True
        elif self.config.use_lemmatization:
            # Lemmatization: prefer spaCy if available, otherwise WordNet
            if self.config.use_spacy_tokenizer:
                use_spacy_lemma = True
            else:
                use_wordnet_lemma = True

        if self.config.use_spacy_tokenizer:
            self.tokenizer = SpacyTokenizer(use_lemmatization=use_spacy_lemma)
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

        # Initialize lemmatizer or stemmer (mutually exclusive)
        self.lemmatizer = None
        self.stemmer = None

        if use_porter_stem:
            self.stemmer = PorterStemmerWrapper()
        elif use_wordnet_lemma:
            self.lemmatizer = WordNetLemmatizerWrapper()

    def preprocess(self, text: str) -> list[str]:
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text received")
            return []

        text = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(text)
        tokens = self.filter.filter_tokens(tokens)

        # Apply stemmer or lemmatizer (one or none, never both)
        if self.stemmer:
            tokens = self.stemmer.stem(tokens)
        elif self.lemmatizer:
            tokens = self.lemmatizer.lemmatize(tokens)

        return tokens

    def preprocess_batch(self, texts: list[str], batch_size: Optional[int] = None) -> list[list[str]]:
        """
        Preprocess a batch of texts, preserving order and blank-vs-non-blank
        alignment. One spaCy streaming pass (nlp.pipe) instead of per-text
        pipeline calls, which matters when indexing whole documents.
        """
        if not texts:
            return []
        if not hasattr(self.tokenizer, "tokenize_many"):
            return [self.preprocess(text) for text in texts]

        normalized = [self.normalizer.normalize(text) for text in texts]
        token_batches = self.tokenizer.tokenize_many(normalized, batch_size=batch_size)

        results = []
        for tokens in token_batches:
            tokens = self.filter.filter_tokens(tokens)
            if self.stemmer:
                tokens = self.stemmer.stem(tokens)
            elif self.lemmatizer:
                tokens = self.lemmatizer.lemmatize(tokens)
            results.append(tokens)
        return results


_default_preprocessor: Optional[TextPreprocessor] = None

# Cache of preprocessors keyed by config, so repeated API calls with the same
# options do not rebuild filters/sets on every request. Bounded LRU.
_config_preprocessor_cache: "OrderedDict[tuple, TextPreprocessor]" = OrderedDict()
_cache_lock = Lock()


def _config_cache_key(config: PreprocessingConfig) -> tuple:
    # to_dict() exposes custom_stopwords as a list; sets/lists/sortables are
    # normalized to sorted tuples so the key is stable and hashable
    return tuple(
        sorted(
            (key, tuple(sorted(value)) if isinstance(value, (set, list, tuple)) else value)
            for key, value in config.to_dict().items()
        )
    )


def get_default_preprocessor() -> TextPreprocessor:
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = TextPreprocessor()
    return _default_preprocessor


def get_preprocessor(config: Optional[PreprocessingConfig] = None) -> TextPreprocessor:
    """Return a (cached) preprocessor for the given config; default when None."""
    if config is None:
        return get_default_preprocessor()

    key = _config_cache_key(config)
    with _cache_lock:
        cached = _config_preprocessor_cache.get(key)
        if cached is not None:
            _config_preprocessor_cache.move_to_end(key)
            return cached

    preprocessor = TextPreprocessor(config)
    with _cache_lock:
        _config_preprocessor_cache[key] = preprocessor
        if len(_config_preprocessor_cache) > _PREPROCESSOR_CACHE_SIZE:
            _config_preprocessor_cache.popitem(last=False)
    return preprocessor


def preprocess(text: str, config: Optional[PreprocessingConfig] = None) -> list[str]:
    return get_preprocessor(config).preprocess(text)


def preprocess_batch(
    texts: list[str],
    config: Optional[PreprocessingConfig] = None,
    batch_size: Optional[int] = None,
) -> list[list[str]]:
    """Preprocess many texts with one (cached) preprocessor."""
    return get_preprocessor(config).preprocess_batch(texts, batch_size=batch_size)
