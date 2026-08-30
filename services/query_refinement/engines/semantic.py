from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import numpy as np
import logging
from typing import List, Set
from collections import Counter
from common.device import get_torch_device
from common.config import get_config

logger = logging.getLogger(__name__)


class SemanticRefinementEngine:
    def __init__(self, model_name: str = None):
        device = get_torch_device()
        if model_name is None:
            model_name = get_config("models.embedding.model_name", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=False)
        self._embedding_cache = {}
        self._cache_max_size = 1000
        logger.info(f"Loaded semantic model: {model_name} on {device}")
    
    def compute_similarity(self, query: str, history: List[str]) -> float:
        if not query or not query.strip() or not history:
            return 0.0
        
        try:
            query_embedding = self._get_embedding_cached(query)
            
            recent_history = history[-3:]
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
        threshold: float = 0.35
    ) -> List[str]:
        if not history or similarity < threshold:
            logger.debug(f"History expansion skipped: similarity={similarity:.3f} < {threshold}")
            return []
        
        query_words = set(word.lower() for word in query.split() if word.isalpha())
        stop_words = self._get_stopwords()
        
        history_word_freq = Counter()
        for hist_query in history[-3:]:
            for word in hist_query.lower().split():
                if word.isalpha() and len(word) > 3 and word not in stop_words:
                    history_word_freq[word] += 1
        
        relevant_terms = []
        for word, freq in history_word_freq.most_common():
            if word not in query_words and freq >= 2:
                relevant_terms.append(word)
                if len(relevant_terms) >= 2:
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
