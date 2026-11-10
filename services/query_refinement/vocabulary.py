"""The dataset's own vocabulary, used to keep the spell checker away from domain terms.

Cranfield queries contain "pitot", "planform" and "mach". The English dictionary does not know them and rewrote them
to "pilot", "platform" and "much" - a measurable loss on the retrieval metrics. When a request names a dataset, the
terms of its TF-IDF index are read once and cached, and a word the corpus knows is never "corrected".
"""
from __future__ import annotations

import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Set

from common.config import get_dataset_specific_path

logger = logging.getLogger(__name__)


class DatasetVocabulary:
    """Cached ``{dataset: words}`` lookup; an unusable dataset simply protects nothing."""

    def __init__(self, cache_size: int = 4, loader: Optional[Callable[[str], Set[str]]] = None):
        self._cache: "OrderedDict[str, Set[str]]" = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self._loader = loader or load_tfidf_vocabulary

    def words(self, dataset: Optional[str]) -> Set[str]:
        if not dataset:
            return set()
        cached = self._cache.get(dataset)
        if cached is not None:
            self._cache.move_to_end(dataset)
            return cached
        try:
            words = self._loader(dataset) or set()
        except Exception as exc:  # noqa: BLE001 - protection is best effort, never a reason to fail a request
            logger.warning("Vocabulary loader failed for %r: %s", dataset, exc)
            words = set()
        self._cache[dataset] = words
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return words


def load_tfidf_vocabulary(dataset: str) -> Set[str]:
    """Lower-cased terms of the dataset's TF-IDF vectorizer, or an empty set when it is not available."""
    try:
        vectorizer_path = Path(get_dataset_specific_path("indices.tfidf_vectorizer", dataset))
    except Exception as exc:  # noqa: BLE001 - a bad dataset name must not fail a refinement request
        logger.warning("Cannot resolve the vocabulary path for %r: %s", dataset, exc)
        return set()

    if not vectorizer_path.is_file():
        logger.info("No TF-IDF index for %r (%s): domain terms are not protected", dataset, vectorizer_path)
        return set()

    try:
        with vectorizer_path.open("rb") as handle:
            vectorizer = pickle.load(handle)
        words = {str(term).lower() for term in vectorizer.vocabulary_}
    except Exception as exc:  # noqa: BLE001 - a broken index only costs the protection
        logger.warning("Cannot read the vocabulary of %r (%s): %s", dataset, vectorizer_path, exc)
        return set()

    logger.info("Loaded %d vocabulary terms for %r", len(words), dataset)
    return words
