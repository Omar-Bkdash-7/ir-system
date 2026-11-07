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
    
    # Ensure NLTK data is available (before loading resources)
    _ensure_nltk_data()
    
    if use_spacy:
        get_spacy_model()
    
    if use_stemming:
        get_stemmer()
    
    if use_lemmatizer:
        get_lemmatizer()
    
    if use_stopwords:
        get_stop_words()
    
    logger.info("All preprocessing resources ready")


def _ensure_nltk_data():
    """
    Make sure NLTK data (wordnet, omw-1.4, stopwords) is available.

    Data already present in the image is verified locally without any network
    access. A partially downloaded/corrupt zip is removed and re-downloaded,
    guarded by a short socket timeout so a dead DNS/network never blocks or
    breaks startup indefinitely. Any problem here is logged and swallowed:
    warmup is a best-effort fallback, the app must still start.
    """
    _required_nltk_data = {
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'stopwords': 'corpora/stopwords',
    }
    missing = [
        pkg for pkg, rel_path in _required_nltk_data.items()
        if not _nltk_path_exists(rel_path)
    ]
    if not missing:
        logger.info("NLTK data already available")
        return

    _remove_corrupt_nltk_files()

    try:
        import nltk
        import socket
        logger.info(f"Downloading missing NLTK data: {missing}...")
        # nltk.download has no timeout parameter; bounded socket timeout avoids
        # blocking startup forever / leaving truncated downloads when the
        # network is unreachable
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(60)
        try:
            for pkg in missing:
                _download_nltk_package(nltk, pkg)
        finally:
            socket.setdefaulttimeout(old_timeout)
    except Exception as e:
        logger.warning(f"Could not download NLTK data during warmup: {e}")


def _nltk_path_exists(rel_path: str) -> bool:
    """
    Check each NLTK search-path directory separately, so one corrupt or missing
    file does not mask the image-cache copy in `nltk.data.path[0]`.

    Uses direct filesystem checks (nltk.data.find mishandles directory entries
    inside zips without a trailing slash, e.g. 'corpora/wordnet', and would
    report the bundled cache as missing).
    """
    import os
    import zipfile
    from nltk.data import path as nltk_data_path

    stem = rel_path.rsplit("/", 1)[-1]
    for data_dir in list(nltk_data_path):
        expanded = os.path.join(data_dir, rel_path)
        if os.path.isdir(expanded):
            return True
        zip_path = expanded + ".zip"
        if os.path.isfile(zip_path):
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    if any(name.startswith(stem + "/") for name in zf.namelist()):
                        return True
            except Exception as e:
                logger.debug(f"Unreadable NLTK bundle '{zip_path}': {e}")
                continue
    return False


def _remove_corrupt_nltk_files():
    """Delete obviously-corrupt (non-zip) NLTK bundles in writable data dirs so they can be re-downloaded."""
    import os
    import glob
    from nltk.data import path as nltk_data_path

    for data_dir in [d for d in nltk_data_path if os.path.isdir(d) and os.access(d, os.W_OK)]:
        for zip_path in glob.glob(os.path.join(data_dir, "corpora", "*.zip")):
            try:
                import zipfile
                with zipfile.ZipFile(zip_path) as zf:
                    bad = zf.testzip()
                if bad is not None:
                    logger.warning(f"Removing corrupt NLTK bundle: {zip_path} (bad file: {bad})")
                    os.remove(zip_path)
            except Exception as e:
                logger.warning(f"Removing unreadable NLTK bundle: {zip_path} ({e})")
                try:
                    os.remove(zip_path)
                except OSError as rm_err:
                    logger.warning(f"Could not remove {zip_path}: {rm_err}")


def _download_nltk_package(nltk, pkg: str):
    """Download one NLTK package, tolerating network failures and corrupt results."""
    try:
        nltk.download(pkg, quiet=True, raise_on_error=False)
    except Exception as e:
        logger.warning(f"Could not download NLTK package '{pkg}': {e}")
