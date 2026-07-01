import threading
import logging
import time
from typing import Optional
import spacy
from spacy.language import Language
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from common.exceptions import ModelLoadError
from .exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)

_nlp: Optional[Language] = None
_stemmer: Optional[PorterStemmer] = None
_lemmatizer: Optional[WordNetLemmatizer] = None
_stop_words: Optional[set[str]] = None

_nlp_lock = threading.Lock()
_stemmer_lock = threading.Lock()
_lemmatizer_lock = threading.Lock()
_stop_words_lock = threading.Lock()


def get_spacy_model() -> Language:
    global _nlp
    
    if _nlp is not None:
        return _nlp
    
    with _nlp_lock:
        if _nlp is not None:
            return _nlp
        
        try:
            logger.info("Loading spaCy model 'en_core_web_sm'...")
            start_time = time.time()
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            elapsed = time.time() - start_time
            logger.info(f"spaCy model loaded in {elapsed:.2f}s")
            return _nlp
        except OSError as e:
            error_msg = (
                "spaCy model 'en_core_web_sm' not found. "
                "Rebuild the Docker image or verify the model download stage."
            )
            logger.error(f"{error_msg} Original error: {e}")
            raise ModelLoadError(error_msg) from e
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise ModelLoadError(f"Failed to load spaCy model: {e}") from e


def get_stemmer() -> PorterStemmer:
    global _stemmer
    
    if _stemmer is not None:
        return _stemmer
    
    with _stemmer_lock:
        if _stemmer is not None:
            return _stemmer
        
        try:
            logger.info("Loading Porter Stemmer...")
            _stemmer = PorterStemmer()
            logger.info("Porter Stemmer loaded")
            return _stemmer
        except Exception as e:
            logger.error(f"Failed to load Porter Stemmer: {e}")
            raise ModelLoadError(f"Failed to load Porter Stemmer: {e}") from e


def get_lemmatizer() -> WordNetLemmatizer:
    global _lemmatizer
    
    if _lemmatizer is not None:
        return _lemmatizer
    
    with _lemmatizer_lock:
        if _lemmatizer is not None:
            return _lemmatizer
        
        try:
            logger.info("Loading WordNet Lemmatizer...")
            start_time = time.time()
            _lemmatizer = WordNetLemmatizer()
            elapsed = time.time() - start_time
            logger.info(f"WordNet Lemmatizer loaded in {elapsed:.2f}s")
            return _lemmatizer
        except LookupError as e:
            error_msg = (
                "WordNet resources not found. "
                "Rebuild the Docker image or verify NLTK data download."
            )
            logger.error(f"{error_msg} Original error: {e}")
            raise ResourceNotFoundError(error_msg) from e
        except Exception as e:
            logger.error(f"Failed to load WordNet Lemmatizer: {e}")
            raise ModelLoadError(f"Failed to load WordNet Lemmatizer: {e}") from e


def get_stop_words() -> set[str]:
    global _stop_words
    
    if _stop_words is not None:
        return _stop_words
    
    with _stop_words_lock:
        if _stop_words is not None:
            return _stop_words
        
        try:
            logger.info("Loading NLTK stopwords...")
            start_time = time.time()
            _stop_words = set(stopwords.words("english"))
            elapsed = time.time() - start_time
            logger.info(f"Stopwords loaded ({len(_stop_words)} words) in {elapsed:.2f}s")
            return _stop_words
        except LookupError as e:
            error_msg = (
                "NLTK stopwords not found. "
                "Rebuild the Docker image or verify NLTK data download."
            )
            logger.error(f"{error_msg} Original error: {e}")
            raise ResourceNotFoundError(error_msg) from e
        except Exception as e:
            logger.error(f"Failed to load stopwords: {e}")
            raise ModelLoadError(f"Failed to load stopwords: {e}") from e


def warmup_resources(use_spacy: bool = True, use_stemming: bool = False, 
                     use_lemmatizer: bool = False, use_stopwords: bool = True):
    logger.info("Warming up preprocessing resources...")
    
    if use_spacy:
        get_spacy_model()
    
    if use_stemming:
        get_stemmer()
    
    if use_lemmatizer:
        get_lemmatizer()
    
    if use_stopwords:
        get_stop_words()
    
    logger.info("All preprocessing resources ready")
