"""The refinement pipeline: spelling -> synonym expansion -> history terms -> anti-drift gate.

Each stage is optional and independent. A missing WordNet corpus or embedding model disables exactly that stage and
leaves the rest working - that is what the previous implementation got wrong: an exception while initialising NLTK
took down the singleton, so every request failed with a 500 instead of answering with the stages that did work.

The result is a plain dictionary that matches :class:`services.query_refinement.schemas.RefineResponse`; the HTTP
layer only converts it, so the pipeline stays testable without a server.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Sequence

from common.config import get_config
from services.query_refinement.corpora import NltkCorpora
from services.query_refinement.engines import (
    QueryExpansionEngine,
    SemanticRefinementEngine,
    SpellingCorrectionEngine,
)
from services.query_refinement.vocabulary import DatasetVocabulary

logger = logging.getLogger(__name__)


class QueryRefinementService:
    def __init__(
        self,
        corpora: Optional[NltkCorpora] = None,
        spelling_engine: Optional[SpellingCorrectionEngine] = None,
        expansion_engine: Optional[QueryExpansionEngine] = None,
        semantic_engine: Optional[SemanticRefinementEngine] = None,
        vocabulary: Optional[DatasetVocabulary] = None,
    ):
        self.vocabulary = vocabulary or DatasetVocabulary()
        self.corpora = corpora or NltkCorpora(
            allow_download=bool(get_config("query_refinement.allow_nltk_download", False))
        )
        stop_words = self.corpora.stopwords()

        self.spelling_engine = spelling_engine or SpellingCorrectionEngine(
            max_edit_distance=get_config("query_refinement.max_edit_distance", 2),
            confidence_threshold=get_config("query_refinement.confidence_threshold", 0.7),
            stop_words=stop_words,
            min_correction_frequency=get_config("query_refinement.min_correction_frequency", 1000),
            min_single_edit_frequency=get_config("query_refinement.min_single_edit_frequency", 100),
        )
        self.expansion_engine = expansion_engine or QueryExpansionEngine(
            max_synonyms_per_word=get_config("query_refinement.max_synonyms_per_word", 2),
            max_synsets=get_config("query_refinement.max_wordnet_synsets", 2),
            min_term_length=get_config("query_refinement.min_term_length", 4),
            wordnet_reader=self.corpora.wordnet(),
            stop_words=stop_words,
            max_query_words=get_config("query_refinement.max_query_words_for_expansion", 4),
            min_synonym_count=get_config("query_refinement.min_synonym_count", 2),
        )
        self.semantic_engine = semantic_engine or SemanticRefinementEngine(stop_words=stop_words)

        self.similarity_threshold = get_config("query_refinement.similarity_threshold", 0.35)
        self.max_drift = get_config("query_refinement.max_semantic_drift", 0.25)
        self.history_boost_weight = get_config("query_refinement.history_boost_weight", 0.15)
        self.max_history_terms = get_config("query_refinement.max_history_terms", 3)

        wordnet = self.corpora.status().get("wordnet")
        logger.info(
            "QueryRefinementService ready (wordnet=%s, expansion=%s, embeddings=%s)",
            bool(wordnet and wordnet.available),
            self.expansion_engine.available,
            self.semantic_engine.available,
        )

    # -- status ------------------------------------------------------------
    def feature_status(self) -> Dict[str, object]:
        """What actually works right now, for /health and the refinement lab."""
        corpora = self.corpora.status()
        wordnet = corpora.get("wordnet")
        return {
            "features": {
                "spelling_correction": True,
                "synonym_expansion": bool(wordnet and wordnet.available),
                "history_boosting": self.semantic_engine.available,
                "semantic_similarity": self.semantic_engine.available,
            },
            "corpora": {name: status.as_dict() for name, status in corpora.items()},
            "embeddings_loaded": self.semantic_engine.model_loaded,
        }

    # -- pipeline ----------------------------------------------------------
    def refine(
        self,
        query: str,
        history: Sequence[str] = (),
        enabled: bool = True,
        session_id: str = "default",
        is_evaluation: bool = False,
        dataset: Optional[str] = None,
    ) -> Dict[str, object]:
        """Run the pipeline and return a result matching RefineResponse."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")

        if not enabled:
            logger.info("Refinement disabled for session %s", session_id)
            return self._result(query, query, applied=False)

        # Only the history the caller sends is used. Keeping a per-session store in the service meant independent
        # queries of an evaluation run (all on the default session, with no history) borrowed terms from each other.
        session_history = [item.strip() for item in history or () if isinstance(item, str) and item.strip()]
        logger.info(
            "Refining %r (session=%s, history=%d, evaluation=%s)",
            query[:120], session_id, len(session_history), is_evaluation,
        )

        corrected, corrections = self.spelling_engine.correct(query, self.vocabulary.words(dataset))
        changes: List[str] = ["spelling"] if corrections else []

        if is_evaluation:
            return self._result(
                query, corrected,
                corrections=corrections,
                changes=changes,
                confidence=1.0 if corrections else 0.0,
                applied=corrected != query,
                evaluation_mode=True,
            )

        expanded, expansions = self.expansion_engine.expand(corrected)
        if expansions:
            changes.append("expansion")

        history_terms: List[str] = []
        similarity = 0.0
        if session_history:
            similarity = self.semantic_engine.compute_similarity(query, session_history)
            if similarity >= self.similarity_threshold:
                history_terms = self.semantic_engine.extract_relevant_history_terms(
                    corrected,
                    session_history,
                    similarity,
                    threshold=self.similarity_threshold,
                    max_terms=self.max_history_terms,
                )
        history_boosts = {term: self.history_boost_weight for term in history_terms}
        if history_terms:
            changes.append("history")

        refined = " ".join([expanded, *history_terms]).strip() if history_terms else expanded

        drift = 0.0
        drift_rejected = False
        if expansions or history_terms:
            acceptable, drift = self.semantic_engine.validate_semantic_drift(
                corrected, refined, max_drift=self.max_drift
            )
            if not acceptable:
                # Keep the deterministic part (spelling) and drop everything that moved the meaning.
                refined = corrected
                expansions, history_boosts, history_terms = {}, {}, []
                changes = ["spelling"] if corrections else []
                drift_rejected = True

        result = self._result(
            query, refined,
            corrections=corrections,
            expansions=expansions,
            history_boosts=history_boosts,
            history_terms=history_terms,
            history_similarity=similarity,
            changes=changes,
            confidence=self._confidence(corrections, expansions, history_boosts, drift),
            semantic_drift=drift,
            applied=refined != query,
            drift_rejected=drift_rejected,
        )

        logger.info(
            "Refined %r -> %r (changes=%s, confidence=%.2f, drift=%.3f)",
            query[:80], refined[:80], changes, result["confidence"], drift,
        )
        return result

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _confidence(
        corrections: Dict[str, str],
        expansions: Dict[str, List[str]],
        history_boosts: Dict[str, float],
        drift: float,
    ) -> float:
        """Evidence-based score: each applied stage contributes, measured drift erodes the total."""
        evidence = 0.0
        if corrections:
            evidence += 0.4
        if expansions:
            evidence += 0.3
        if history_boosts:
            evidence += 0.3
        if evidence == 0.0:
            return 0.0
        return round(min(1.0, evidence * (1.0 - min(0.5, max(0.0, drift)))), 3)

    @staticmethod
    def _result(
        original_query: str,
        refined_query: str,
        *,
        corrections: Optional[Dict[str, str]] = None,
        expansions: Optional[Dict[str, List[str]]] = None,
        history_boosts: Optional[Dict[str, float]] = None,
        history_terms: Optional[List[str]] = None,
        history_similarity: float = 0.0,
        changes: Optional[List[str]] = None,
        confidence: float = 0.0,
        semantic_drift: float = 0.0,
        applied: bool = True,
        evaluation_mode: bool = False,
        drift_rejected: bool = False,
    ) -> Dict[str, object]:
        return {
            "original_query": original_query,
            "refined_query": refined_query,
            "corrections": dict(corrections or {}),
            "expansions": dict(expansions or {}),
            "history_boosts": dict(history_boosts or {}),
            "history_terms": list(history_terms or []),
            "history_similarity": round(float(history_similarity), 4),
            "confidence": confidence,
            "changes_applied": list(changes or []),
            "semantic_drift": semantic_drift,
            "applied": applied,
            "evaluation_mode": evaluation_mode,
            "drift_rejected": drift_rejected,
        }


_service: Optional[QueryRefinementService] = None
_service_lock = threading.Lock()


def get_service() -> QueryRefinementService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = QueryRefinementService()
    return _service


def refine_query(
    query: str,
    history: Sequence[str] = (),
    enabled: bool = True,
    session_id: str = "default",
    is_evaluation: bool = False,
    dataset: Optional[str] = None,
) -> Dict[str, object]:
    """Refine ``query`` with the shared service instance."""
    return get_service().refine(query, history, enabled, session_id, is_evaluation, dataset)
