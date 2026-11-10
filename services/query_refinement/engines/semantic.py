"""Embedding-backed history similarity and drift checks.

The model is loaded on first use instead of in the constructor: the old constructor loaded it during the
singleton's initialisation, so an unavailable model meant the service could not answer anything, including
requests that only needed spelling. Every method now degrades to a neutral answer (no similarity, no drift) when the
model is missing; rejecting every refinement because a model failed to load is worse than skipping the check.
"""
from __future__ import annotations

import logging
from collections import OrderedDict, Counter
from typing import List, Optional, Set, Tuple

import numpy as np

from common.device import get_torch_device
from common.model_loader import load_embedding_model

logger = logging.getLogger(__name__)

# Only the most recent queries describe the session; older ones add noise, not context.
HISTORY_WINDOW = 5
CACHE_SIZE = 512


class SemanticRefinementEngine:
    def __init__(
        self,
        model=None,
        model_loader=None,
        stop_words: Optional[Set[str]] = None,
        cache_size: int = CACHE_SIZE,
    ):
        self._model = model
        self._model_loader = model_loader or load_embedding_model
        self._load_failed = False
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size
        self.stop_words = set(stop_words) if stop_words is not None else set()

    # -- model -------------------------------------------------------------
    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def available(self) -> bool:
        """True when embeddings can still be produced (loaded already, or not yet attempted and not known broken)."""
        return self._model is not None or not self._load_failed

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            device = get_torch_device()
            self._model = self._model_loader(device=device, cache=True)
            logger.info("Semantic refinement model ready on %s", device)
        except Exception as exc:  # noqa: BLE001 - a missing model disables a stage, it must not fail the service
            self._load_failed = True
            logger.warning("Embedding model unavailable (%s): history boosting and drift checks are disabled", exc)
        return self._model

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Normalised embedding of ``text``, or None when the model or the encoding failed."""
        if not text or not text.strip():
            return None
        model = self._ensure_model()
        if model is None:
            return None

        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached

        try:
            vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding failed for %r: %s", text[:80], exc)
            return None

        vector = np.asarray(vector, dtype=np.float32)
        self._cache[text] = vector
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vector

    # -- similarity --------------------------------------------------------
    def compute_similarity(self, query: str, history: List[str]) -> float:
        """Best cosine similarity between the query and the last few queries of the session.

        The best match is more informative than the embedding of the concatenated history: one related earlier
        query is a real signal, while averaging it with unrelated queries drowns it.
        """
        if not query or not query.strip() or not history:
            return 0.0
        query_vector = self.encode(query)
        if query_vector is None:
            return 0.0

        best = 0.0
        for previous in history[-HISTORY_WINDOW:]:
            vector = self.encode(previous)
            if vector is None:
                continue
            best = max(best, float(np.dot(query_vector, vector)))
        return round(max(0.0, min(1.0, best)), 4)

    def extract_relevant_history_terms(
        self,
        query: str,
        history: List[str],
        similarity: float,
        threshold: float = 0.25,
        max_terms: int = 3,
    ) -> List[str]:
        """Terms repeated in the session that the current query lacks, most frequent first."""
        if not history or similarity < threshold:
            logger.debug("History terms skipped: similarity %.3f < %.3f", similarity, threshold)
            return []

        query_words = {word.lower() for word in query.split() if word.isalpha()}
        frequency: Counter = Counter()
        for previous in history[-HISTORY_WINDOW:]:
            for word in previous.lower().split():
                if word.isalpha() and len(word) > 3 and word not in self.stop_words:
                    frequency[word] += 1

        terms = [
            word for word, count in frequency.most_common()
            if count >= 2 and word not in query_words
        ][:max_terms]
        if terms:
            logger.info("History terms for %r: %s", query[:60], terms)
        return terms

    # -- drift -------------------------------------------------------------
    def validate_semantic_drift(
        self,
        original_query: str,
        refined_query: str,
        max_drift: float = 0.25,
    ) -> Tuple[bool, float]:
        """Return ``(is_acceptable, drift)`` where drift is ``1 - cosine`` between the two queries.

        Without embeddings the drift cannot be measured, and the refinement is accepted: the stages that produce it
        are conservative on their own, and refusing everything would make the feature useless on a model-less host.
        """
        if not original_query or not refined_query or original_query == refined_query:
            return True, 0.0

        original_vector = self.encode(original_query)
        refined_vector = self.encode(refined_query)
        if original_vector is None or refined_vector is None:
            return True, 0.0

        drift = round(max(0.0, 1.0 - float(np.dot(original_vector, refined_vector))), 4)
        acceptable = drift <= max_drift
        if acceptable:
            logger.debug("Semantic drift acceptable: %.3f <= %.2f", drift, max_drift)
        else:
            logger.info(
                "Semantic drift too high: %.3f > %.2f (%r -> %r)",
                drift, max_drift, original_query, refined_query,
            )
        return acceptable, drift
