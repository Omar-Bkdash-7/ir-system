from nltk.corpus import stopwords
import numpy as np
import logging
from typing import List, Set, Tuple
from collections import Counter
from common.device import get_torch_device
from common.config import get_config
from common.model_loader import load_embedding_model

logger = logging.getLogger(__name__)


class SemanticRefinementEngine:
    def __init__(self, model_name: str = None):
        device = get_torch_device()
        
        if model_name is not None:
            logger.warning(
                "Custom model_name parameter is deprecated. "
                "Model is loaded from centralized config."
            )
        
        logger.info("Loading embedding model using centralized loader")
        self.model = load_embedding_model(device=device, cache=True)
        
        self._embedding_cache = {}
        self._cache_max_size = 1000
        logger.info(f"Semantic refinement engine initialized on {device}")
    
    def compute_similarity(self, query: str, history: List[str]) -> float:
        """
        Compute semantic similarity between query and recent history.
        Uses last 5 queries for better context.
        """
        if not query or not query.strip() or not history:
            return 0.0
        
        try:
            query_embedding = self._get_embedding_cached(query)
            
            # Use last 5 queries (not just 3) for richer history
            recent_history = history[-5:]
            history_text = " ".join(recent_history)
            history_embedding = self._get_embedding_cached(history_text)
            
            similarity = float(np.dot(query_embedding, history_embedding))
            return max(0.0, min(1.0, similarity))
        
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.0
    
    def _get_embedding_cached(self, text: str) -> np.ndarray:
        if text not in self._embedding_cache:
            if len(self._embedding_cache) >= self._cache_max_size:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            
            embedding = self.model.encode(
                [text], 
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]
            self._embedding_cache[text] = embedding
        
        return self._embedding_cache[text]

    def extract_relevant_history_terms(
        self, 
        query: str, 
        history: List[str], 
        similarity: float,
        threshold: float = 0.25
    ) -> List[str]:
        """
        Extract relevant terms from history with frequency-based filtering.
        Lower threshold (0.25) to be more inclusive.
        """
        if not history or similarity < threshold:
            logger.debug(f"History expansion skipped: similarity={similarity:.3f} < {threshold}")
            return []
        
        query_words = set(word.lower() for word in query.split() if word.isalpha())
        stop_words = self._get_stopwords()
        
        history_word_freq = Counter()
        for hist_query in history[-5:]:  # Use last 5 queries (not just 3)
            for word in hist_query.lower().split():
                if word.isalpha() and len(word) > 3 and word not in stop_words:
                    history_word_freq[word] += 1
        
        relevant_terms = []
        for word, freq in history_word_freq.most_common():
            # Include terms that appear at least twice
            if word not in query_words and freq >= 2:
                relevant_terms.append(word)
                if len(relevant_terms) >= 3:  # Max 3 history terms
                    break
        
        if relevant_terms:
            logger.info(f"Extracted {len(relevant_terms)} relevant terms from history: {relevant_terms}")
        
        return relevant_terms
    
    @staticmethod
    def _get_stopwords() -> Set[str]:
        try:
            return set(stopwords.words("english"))
        except:
            return set()

    
    def validate_semantic_drift(
        self, 
        original_query: str, 
        refined_query: str, 
        max_drift: float = 0.25
    ) -> Tuple[bool, float]:
        """
        Validate that refined query hasn't drifted too far from original.
        
        Args:
            original_query: Original query text
            refined_query: Refined query text
            max_drift: Maximum allowed drift (default 0.25)
            
        Returns:
            Tuple of (is_valid, drift_score)
        """
        if not original_query or not refined_query:
            return True, 0.0
        
        # If queries are identical, no drift
        if original_query == refined_query:
            return True, 0.0
        
        try:
            original_emb = self._get_embedding_cached(original_query)
            refined_emb = self._get_embedding_cached(refined_query)
            
            # Compute cosine similarity (higher is better)
            similarity = float(np.dot(original_emb, refined_emb))
            
            # Drift is inverse of similarity
            drift = 1.0 - similarity
            
            is_valid = drift <= max_drift
            
            if not is_valid:
                logger.warning(
                    f"Semantic drift too high: {drift:.3f} > {max_drift} "
                    f"(original: '{original_query}', refined: '{refined_query}')"
                )
            else:
                logger.debug(f"Semantic drift acceptable: {drift:.3f} <= {max_drift}")
            
            return is_valid, drift
            
        except Exception as e:
            logger.error(f"Drift validation failed: {e}")
            # On error, reject the refinement to be safe
            return False, 1.0
