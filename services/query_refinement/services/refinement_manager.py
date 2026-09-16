from services.query_refinement.engines import (
    SpellingCorrectionEngine,
    QueryExpansionEngine,
    SemanticRefinementEngine,
    QueryHistoryManager
)
from common.config import get_config
import logging
import nltk
import os
import shutil
from typing import List, Dict, Optional
from nltk.corpus import wordnet, stopwords

logger = logging.getLogger(__name__)


class QueryRefinementService:
    _instance: Optional['QueryRefinementService'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._ensure_nltk_resources()
        
        max_edit_distance = get_config("query_refinement.max_edit_distance", 2)
        confidence_threshold = get_config("query_refinement.confidence_threshold", 0.7)
        self.spelling_engine = SpellingCorrectionEngine(max_edit_distance, confidence_threshold)
        
        max_synonyms = get_config("query_refinement.max_synonyms_per_word", 2)
        max_synsets = get_config("query_refinement.max_wordnet_synsets", 2)
        min_term_length = get_config("query_refinement.min_term_length", 4)
        self.expansion_engine = QueryExpansionEngine(
            max_synonyms, 
            max_synsets, 
            min_term_length
        )
        
        self.semantic_engine = SemanticRefinementEngine()
        
        max_history_length = get_config("query_refinement.max_history_length", 10)
        self.history_manager = QueryHistoryManager(max_history_length)
        
        self.similarity_threshold = get_config("query_refinement.similarity_threshold", 0.25)  # Lowered from 0.35
        
        self._initialized = True
        logger.info("QueryRefinementService initialized successfully")

    def _ensure_nltk_resources(self) -> None:
        nltk_home = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_home, exist_ok=True)
        
        if nltk_home not in nltk.data.path:
            nltk.data.path.insert(0, nltk_home)
        
        resources = ['wordnet', 'stopwords', 'omw-1.4']
        
        for resource in resources:
            downloaded = False
            for attempt in range(3):
                try:
                    if resource == 'wordnet':
                        wordnet.synsets('test')
                        downloaded = True
                        logger.info(f"NLTK '{resource}' is ready")
                        break
                    elif resource == 'stopwords':
                        stopwords.words('english')
                        downloaded = True
                        logger.info(f"NLTK '{resource}' is ready")
                        break
                    elif resource == 'omw-1.4':
                        try:
                            nltk.data.find('corpora/omw-1.4')
                            downloaded = True
                            logger.info(f"NLTK '{resource}' is ready")
                            break
                        except LookupError:
                            pass
                except (LookupError, OSError):
                    logger.info(f"Downloading NLTK '{resource}' (attempt {attempt + 1}/3)...")
                    
                    if attempt > 0:
                        for subdir in ['corpora', 'tokenizers', 'taggers']:
                            resource_path = os.path.join(nltk_home, subdir, resource)
                            if os.path.exists(resource_path):
                                shutil.rmtree(resource_path, ignore_errors=True)
                            
                            zip_path = resource_path + '.zip'
                            if os.path.exists(zip_path):
                                try:
                                    os.remove(zip_path)
                                except:
                                    pass
                    
                    try:
                        nltk.download(resource, download_dir=nltk_home, quiet=False)
                    except Exception as download_error:
                        logger.warning(f"Download error for '{resource}': {download_error}")
                        if attempt == 2:
                            logger.error(f"Failed to download '{resource}' after 3 attempts")
            
            if not downloaded and resource in ['wordnet', 'stopwords']:
                logger.warning(f"Could not verify '{resource}', will use fallback methods")

    def refine(
        self, 
        query: str, 
        history: List[str], 
        enabled: bool,
        session_id: str = "default",
        is_evaluation: bool = False
    ) -> Dict:
        """
        Refine query using multi-stage pipeline.
        
        Args:
            query: Original query text
            history: Query history (for interactive sessions)
            enabled: Whether refinement is enabled
            session_id: Session identifier
            is_evaluation: If True, only apply spelling correction (deterministic)
            
        Returns:
            Dictionary with refinement results and metadata
        """
        if not enabled:
            logger.info("Query refinement disabled")
            return self._empty_result(query, applied=False)
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(
            f"Refining query: '{query}' | Session: {session_id} | "
            f"History: {len(history)} queries | Evaluation mode: {is_evaluation}"
        )
        
        changes_applied = []
        
        # ===== STAGE 1: Spelling Correction (always safe) =====
        corrected_query, corrections = self.spelling_engine.correct(query)
        
        if corrections:
            changes_applied.append("spelling")
            logger.info(f"Spelling corrections applied: {corrections}")
        
        # ===== EVALUATION MODE: Stop here (deterministic only) =====
        if is_evaluation:
            logger.info("Evaluation mode: applying spelling correction only")
            return {
                "original_query": query,
                "refined_query": corrected_query,
                "corrections": corrections,
                "expansions": {},
                "history_boosts": {},
                "confidence": 1.0 if corrections else 0.0,
                "changes_applied": changes_applied,
                "semantic_drift": 0.0,
                "applied": True,
                "evaluation_mode": True
            }
        
        # ===== STAGE 2: Query Expansion (careful synonym addition) =====
        expanded_query = corrected_query
        expansions = {}
        
        # Only expand if confidence is high enough
        query_words = corrected_query.lower().split()
        if len(query_words) <= 4:  # Only expand short-ish queries
            expanded_query, expansions = self.expansion_engine.expand(corrected_query)
            
            if expansions:
                changes_applied.append("expansion")
                logger.info(f"Query expanded with synonyms: {expansions}")
        else:
            logger.debug(f"Skipping expansion: query has {len(query_words)} words")
        
        # ===== STAGE 3: History-based Term Boosting (interactive sessions) =====
        history_boosts = {}
        history_terms = []
        similarity = 0.0
        
        if history and len(history) >= 2:
            self.history_manager.add_query(session_id, corrected_query)
            similarity = self.semantic_engine.compute_similarity(query, history)
            logger.debug(f"History similarity: {similarity:.3f}")
            
            # Lower threshold (0.25) to be more inclusive
            if similarity > 0.25:
                history_terms = self.semantic_engine.extract_relevant_history_terms(
                    corrected_query,
                    history,
                    similarity,
                    threshold=0.25
                )
                
                if history_terms:
                    changes_applied.append("history")
                    # Assign boost weight of 0.15 to each history term
                    history_boosts = {term: 0.15 for term in history_terms}
                    logger.info(f"History terms boosted: {history_boosts}")
        
        # Build final query
        final_query = expanded_query
        if history_terms:
            final_query = expanded_query + " " + " ".join(history_terms)
        
        # ===== STAGE 4: Anti-drift Validation =====
        # Only validate drift if we did expansion or history (not just spelling)
        drift_score = 0.0
        
        if expansions or history_terms:
            is_valid, drift_score = self.semantic_engine.validate_semantic_drift(
                corrected_query,  # Compare against corrected query, not original
                final_query,
                max_drift=0.25
            )
            
            if not is_valid:
                logger.warning(
                    f"Refinement rejected due to excessive semantic drift: {drift_score:.3f}"
                )
                # Fall back to corrected query (keep spelling, reject expansion/history)
                return {
                    "original_query": query,
                    "refined_query": corrected_query,
                    "corrections": corrections,
                    "expansions": {},
                    "history_boosts": {},
                    "confidence": 0.5,
                    "changes_applied": ["spelling"] if corrections else [],
                    "semantic_drift": drift_score,
                    "applied": True,
                    "drift_rejected": True
                }
        else:
            # No expansion/history, no drift validation needed
            # But still calculate drift for corrected query
            if corrected_query != query:
                _, drift_score = self.semantic_engine.validate_semantic_drift(
                    query,
                    corrected_query,
                    max_drift=1.0  # Allow any drift for spelling only
                )
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(corrections, expansions, history_boosts, drift_score)
        
        # Log final result
        if final_query != query:
            logger.info(
                f"Query refined: '{query}' -> '{final_query}' "
                f"(confidence: {confidence:.2f}, drift: {drift_score:.3f}, changes: {changes_applied})"
            )
        else:
            logger.debug("Query unchanged after refinement")
        
        return {
            "original_query": query,
            "refined_query": final_query,
            "corrections": corrections,
            "expansions": expansions,
            "history_boosts": history_boosts,
            "confidence": confidence,
            "changes_applied": changes_applied,
            "semantic_drift": drift_score,
            "applied": True,
            "evaluation_mode": False
        }
    
    def _empty_result(self, query: str, applied: bool = False) -> Dict:
        """Return empty refinement result."""
        return {
            "original_query": query,
            "refined_query": query,
            "corrections": {},
            "expansions": {},
            "history_boosts": {},
            "confidence": 0.0,
            "changes_applied": [],
            "semantic_drift": 0.0,
            "applied": applied
        }
    
    def _calculate_confidence(
        self,
        corrections: Dict[str, str],
        expansions: Dict[str, List[str]],
        history_boosts: Dict[str, float],
        drift_score: float
    ) -> float:
        """
        Calculate overall confidence score for the refinement.
        
        Returns value between 0.0 and 1.0.
        """
        confidence = 0.0
        
        # Spelling corrections add confidence
        if corrections:
            confidence += 0.3
        
        # Expansions add confidence (but less)
        if expansions:
            confidence += 0.2
        
        # History boosts add confidence
        if history_boosts:
            confidence += 0.2
        
        # Low drift adds confidence
        if drift_score < 0.1:
            confidence += 0.3
        elif drift_score < 0.2:
            confidence += 0.15
        
        return min(1.0, confidence)


_service: Optional[QueryRefinementService] = None


def get_service() -> QueryRefinementService:
    global _service
    if _service is None:
        _service = QueryRefinementService()
    return _service


def refine_query(
    query: str, 
    history: List[str], 
    enabled: bool,
    session_id: str = "default",
    is_evaluation: bool = False
) -> Dict:
    """
    Refine query with optional evaluation mode.
    
    Args:
        query: Query text to refine
        history: List of previous queries
        enabled: Whether refinement is enabled
        session_id: Session identifier
        is_evaluation: If True, only apply deterministic operations (spelling)
    """
    service = get_service()
    return service.refine(query, history, enabled, session_id, is_evaluation)
