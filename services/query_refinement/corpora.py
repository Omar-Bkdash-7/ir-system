"""Loads the NLTK corpora the refinement stages need, without letting a broken corpus break the service.

The corpora ship inside the image under ``NLTK_DATA``. One build baked a truncated ``wordnet.zip`` (10 MB that is
not a zip at all), and ``nltk.data.find`` raised ``BadZipFile`` from inside nltk: the singleton's constructor then
failed, so *every* ``/refine`` call answered 500 even when synonym expansion was not needed.

Loading is now attempted once, verified by actually using the corpus, and cached. A missing or corrupt corpus
disables only the stage that needs it and is reported through :meth:`NltkCorpora.status`.
"""
from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet as nltk_wordnet

logger = logging.getLogger(__name__)

# Used when the stopwords corpus is unavailable: the stages need *some* list, and a fixed one is deterministic.
FALLBACK_STOPWORDS = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "their", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "a", "an", "the", "and", "but", "if", "or", "as", "of", "at", "by", "for", "with", "to", "from", "in", "on",
    "about", "into", "over", "after", "before", "between", "during", "through", "then", "than", "so", "such",
    "no", "not", "only", "own", "same", "too", "very", "can", "will", "just", "should", "now", "what", "which",
    "who", "whom", "this", "that", "these", "those", "there", "here", "how", "when", "where", "why",
})

@dataclass(frozen=True)
class CorpusStatus:
    name: str
    available: bool
    detail: str

    def as_dict(self) -> Dict[str, object]:
        return {"available": self.available, "detail": self.detail}


class NltkCorpora:
    """Verified, cached access to WordNet and the English stopword list.

    ``allow_download`` is off by default: an image is supposed to ship its corpora, and a service that silently
    downloads NLTK data on the first request is slow and depends on network access it may not have.
    """

    def __init__(self, *, allow_download: bool = False, download_dir: Optional[str] = None):
        self._allow_download = allow_download
        self._download_dir = download_dir or os.path.join(os.path.expanduser("~"), "nltk_data")
        self._wordnet = None
        self._stopwords: Optional[Set[str]] = None
        self._status: Dict[str, CorpusStatus] = {}
        self._loaded = False

    # -- public API --------------------------------------------------------
    def wordnet(self):
        """The WordNet reader, or None when the corpus cannot be used."""
        self._load()
        return self._wordnet

    def stopwords(self) -> Set[str]:
        """The English stopword list, or the built-in fallback when the corpus is unusable."""
        self._load()
        return set(self._stopwords or FALLBACK_STOPWORDS)

    def status(self) -> Dict[str, CorpusStatus]:
        self._load()
        return dict(self._status)

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        self._wordnet = self._load_wordnet()
        self._stopwords = self._load_stopwords()
        self._loaded = True

    def _load_wordnet(self):
        try:
            nltk_wordnet.synsets("test")  # a truncated archive fails here, not at import time
        except Exception as exc:  # noqa: BLE001 - nltk raises BadZipFile, OSError, LookupError, ...
            detail = f"{type(exc).__name__}: {exc}"
            logger.warning("WordNet is unusable (%s)", detail)
            if self._repair("wordnet"):
                try:
                    nltk_wordnet.synsets("test")
                except Exception as retry_exc:  # noqa: BLE001
                    detail = f"{type(retry_exc).__name__}: {retry_exc}"
                    logger.warning("WordNet is still unusable after download (%s)", detail)
                else:
                    self._status["wordnet"] = CorpusStatus("wordnet", True, "downloaded")
                    logger.info("WordNet downloaded and verified")
                    return nltk_wordnet
            self._status["wordnet"] = CorpusStatus("wordnet", False, detail)
            logger.warning("Synonym expansion is disabled: WordNet is unavailable")
            return None
        self._status["wordnet"] = CorpusStatus("wordnet", True, "loaded")
        return nltk_wordnet

    def _load_stopwords(self) -> Set[str]:
        try:
            words = set(nltk_stopwords.words("english"))
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            logger.warning("Stopword corpus is unusable (%s), using the built-in list", detail)
            self._status["stopwords"] = CorpusStatus("stopwords", False, f"{detail} (built-in list in use)")
            return set(FALLBACK_STOPWORDS)
        self._status["stopwords"] = CorpusStatus("stopwords", True, "loaded")
        return words

    # -- repair ------------------------------------------------------------
    def _repair(self, name: str) -> bool:
        """Remove a corrupt copy and download the corpus again, but only when that was explicitly allowed."""
        if not self._allow_download:
            logger.warning(
                "Not downloading NLTK '%s': query_refinement.allow_nltk_download is off", name
            )
            return False

        self._drop_corrupt_copies(name)
        download_dir = Path(self._download_dir)
        try:
            download_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Cannot create NLTK download directory %s: %s", download_dir, exc)
            return False
        # nltk.data.find returns the first existing copy, so a repaired one has to be searched first.
        if str(download_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(download_dir))
        try:
            downloaded = nltk.download(name, download_dir=str(download_dir), quiet=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Downloading NLTK '%s' failed: %s", name, exc)
            return False
        if not downloaded:
            logger.warning("NLTK '%s' could not be downloaded", name)
        return bool(downloaded)

    @staticmethod
    def _drop_corrupt_copies(name: str) -> None:
        for base in list(nltk.data.path):
            archive = Path(base) / "corpora" / f"{name}.zip"
            if not archive.is_file() or not _is_corrupt(archive):
                continue
            try:
                archive.unlink()
                logger.warning("Removed corrupt NLTK archive %s", archive)
            except OSError as exc:
                logger.warning("Corrupt NLTK archive %s cannot be removed: %s", archive, exc)


def _is_corrupt(archive: Path) -> bool:
    try:
        with zipfile.ZipFile(archive) as bundle:
            return not bundle.namelist()
    except (zipfile.BadZipFile, OSError):
        return True
