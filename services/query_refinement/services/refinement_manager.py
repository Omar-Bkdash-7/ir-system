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
        
        self.similarity_threshold = get_config("query_refinement.similarity_threshold", 0.35)
        
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
        session_id: str = "default"
    ) -> Dict:
        if not enabled:
            logger.info("Query refinement disabled")
            return {
                "original_query": query,
                "refined_query": query,
                "corrections": {},
                "synonyms_added": {},
                "history_terms_added": [],
                "history_similarity": 0.0,
                "boosted_terms": [],
                "applied": False
            }
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Refining query: '{query}' | Session: {session_id} | History size: {len(history)}")
        
        # CONSERVATIVE APPROACH: Only spelling correction, no expansion
        # Query expansion has been disabled to prevent query drift which degrades performance
        
        # Step 1: Spelling correction (only for obvious typos with high confidence)
        corrected_query, corrections = self.spelling_engine.correct(query)
        
        if corrections:
            logger.info(f"Spelling corrections applied: {corrections}")
        else:
            logger.debug("No spelling corrections needed")
        
        # Step 2: Query expansion is DISABLED
        # Previous implementation added synonyms which caused query drift and degraded metrics
        # For IR evaluation, maintaining query intent is more important than expansion
        logger.debug("Query expansion disabled to maintain query precision and prevent drift")
        
        synonyms_added = {}
        expanded_query = corrected_query
        
        # Step 3: History-based refinement is DISABLED during evaluation
        # History context is not available during batch evaluation
        # Only useful in interactive sessions
        similarity = 0.0
        history_terms = []
        boosted_terms = []
        
        # Only use history in interactive sessions (not default evaluation session)
        if history and session_id != "default":
            self.history_manager.add_query(session_id, corrected_query)
            similarity = self.semantic_engine.compute_similarity(query, history)
            logger.debug(f"History similarity: {similarity:.3f} (not applied during evaluation)")
        
        final_query = corrected_query
        
        # Log refinement result
        if final_query != query:
            logger.info(f"Query refined: '{query}' -> '{final_query}'")
        else:
            logger.debug("Query unchanged after refinement (no corrections needed)")
        
        return {
            "original_query": query,
            "refined_query": final_query,
            "corrections": corrections,
            "synonyms_added": synonyms_added,
            "history_terms_added": history_terms,
            "history_similarity": round(similarity, 3),
            "boosted_terms": boosted_terms,
            "applied": True
        }


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
    session_id: str = "default"
) -> Dict:
    service = get_service()
    return service.refine(query, history, enabled, session_id)
