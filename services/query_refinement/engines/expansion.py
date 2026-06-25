from nltk.corpus import wordnet, stopwords
import numpy as np
import logging
from typing import List, Tuple, Dict, Set

logger = logging.getLogger(__name__)


class QueryExpansionEngine:
    def __init__(
        self, 
        max_synonyms_per_word: int = 2,
        max_synsets: int = 2,
        min_term_length: int = 4
    ):
        self.max_synonyms_per_word = max_synonyms_per_word
        self.max_synsets = max_synsets
        self.min_term_length = min_term_length
        self.stop_words = self._load_stopwords()
        self._synset_cache = {}
    
    def _load_stopwords(self) -> Set[str]:
        try:
            return set(stopwords.words("english"))
        except Exception:
            return set()
    
    def expand(self, query: str) -> Tuple[str, Dict[str, List[str]]]:
        if not query or not query.strip():
            return query, {}
        
        words = query.lower().split()
        original_words_set = set(words)
        synonyms_map = {}
        all_synonyms = []
        
        for word in words:
            if not self._is_expandable(word):
                continue
            
            try:
                synsets = self._get_synsets_cached(word)
                if not synsets:
                    continue
                
                primary_synset = synsets[0]

                word_synonyms = self._extract_ranked_synonyms(
                    word, 
                    synsets[:self.max_synsets],
                    original_words_set,
                    primary_synset
                )
                
                if word_synonyms:
                    selected = word_synonyms[:self.max_synonyms_per_word]
                    synonyms_map[word] = selected
                    all_synonyms.extend(selected)
                    logger.debug(f"Expanded '{word}' with: {selected}")
            
            except Exception as e:
                logger.debug(f"Expansion failed for '{word}': {e}")
                continue
        
        if not all_synonyms:
            return query, {}
        
        expanded_query = query + " " + " ".join(all_synonyms)
        return expanded_query, synonyms_map
    
    def _get_synsets_cached(self, word: str) -> List:
        if word not in self._synset_cache:
            self._synset_cache[word] = wordnet.synsets(word, lang='eng')
        return self._synset_cache[word]
    
    def _is_expandable(self, word: str) -> bool:
        return (
            word.isalpha() and 
            word not in self.stop_words and 
            len(word) >= self.min_term_length
        )
    
    def _extract_ranked_synonyms(
        self, 
        original_word: str, 
        synsets: List, 
        existing_words: Set[str],
        primary_synset
    ) -> List[str]:
        synonym_scores = []
        
        for synset in synsets:
            similarity_to_primary = self._synset_similarity(synset, primary_synset)
            
            for lemma in synset.lemmas():
                lemma_name = lemma.name().replace('_', ' ').lower()

                if self._is_valid_synonym(lemma_name, original_word, existing_words):
                    frequency = lemma.count()
                    score = similarity_to_primary * (1 + np.log1p(frequency))
                    synonym_scores.append((lemma_name, score))
        
        synonym_scores.sort(key=lambda x: x[1], reverse=True)
        return [syn for syn, score in synonym_scores]
    
    def _synset_similarity(self, synset1, synset2) -> float:
        try:
            similarity = synset1.wup_similarity(synset2)
            return similarity if similarity is not None else 0.5
        except:
            return 0.5
    
    def _is_valid_synonym(
        self, 
        synonym: str, 
        original: str, 
        existing_words: Set[str]
    ) -> bool:
        if (
            synonym == original or
            synonym in existing_words or
            synonym in self.stop_words or
            not synonym.isalpha() or
            len(synonym) < 3 or
            ' ' in synonym
        ):
            return False
        
        return True
    
    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        longer = max(len(s1), len(s2))
        if longer == 0:
            return 1.0
        
        s1_set = set(s1)
        s2_set = set(s2)
        intersection = len(s1_set & s2_set)
        union = len(s1_set | s2_set)
        
        return intersection / union if union > 0 else 0.0
