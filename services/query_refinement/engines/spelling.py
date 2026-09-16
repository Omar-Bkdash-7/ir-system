from spellchecker import SpellChecker
from nltk.corpus import stopwords
import logging
from typing import Tuple, Dict, Set

logger = logging.getLogger(__name__)


class SpellingCorrectionEngine:
    def __init__(self, max_edit_distance: int = 2, confidence_threshold: float = 0.7):
        self.spell_checker = SpellChecker()
        self.max_edit_distance = 1  # Only correct single-character typos
        self.confidence_threshold = 0.85  # High confidence required
        self.stop_words = self._load_stopwords()
        self.protected_terms = self._load_protected_terms()
    
    def _load_stopwords(self) -> Set[str]:
        try:
            return set(stopwords.words("english"))
        except Exception as e:
            logger.warning(f"Failed to load NLTK stopwords: {e}. Using fallback.")
            return {
                'i', 'me', 'my', 'we', 'you', 'he', 'she', 'it', 'they',
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the',
                'and', 'but', 'if', 'or', 'as', 'of', 'at', 'by', 'for',
                'with', 'to', 'from', 'in', 'on'
            }
    
    def _load_protected_terms(self) -> Set[str]:
        return {
            'api', 'http', 'https', 'json', 'xml', 'html', 'css', 'sql',
            'nosql', 'mongodb', 'redis', 'kafka', 'nginx', 'docker', 'kubernetes',
            'oauth', 'jwt', 'cors', 'crud', 'rest', 'graphql', 'grpc',
            'ci', 'cd', 'devops', 'mlops', 'ai', 'ml', 'nlp', 'bert',
            'gpu', 'cpu', 'ram', 'ssd', 'raid', 'vpc', 'cdn', 'dns',
            'ssh', 'ftp', 'smtp', 'tcp', 'udp', 'ip', 'lan', 'wan',
            'aws', 'azure', 'gcp', 'github', 'gitlab', 'bitbucket'
        }
    
    def add_domain_terms(self, terms: Set[str]) -> None:
        self.protected_terms.update(term.lower() for term in terms)

    def correct(self, query: str) -> Tuple[str, Dict[str, str]]:
        if not query or not query.strip():
            return query, {}
        
        words = query.lower().split()
        corrected_words = []
        corrections = {}
        
        for word in words:
            # Don't correct: non-alphabetic, stopwords, or very short words
            if not word.isalpha() or word in self.stop_words or len(word) <= 3:
                corrected_words.append(word)
                continue
            
            # Don't correct protected technical/domain terms
            if word in self.protected_terms:
                corrected_words.append(word)
                continue
            
            try:
                # Check if word exists in dictionary first
                if word in self.spell_checker:
                    corrected_words.append(word)
                    continue
                
                # Get correction suggestion
                corrected = self.spell_checker.correction(word)
                
                if corrected and corrected != word:
                    distance = self._levenshtein_distance(word, corrected)
                    
                    # Only accept corrections with distance = 1 (single typo) and very high confidence
                    if distance == 1:
                        confidence = 1.0 - (distance / max(len(word), len(corrected)))
                        
                        # Require very high confidence (0.9+) to avoid over-correction
                        if confidence >= 0.9:
                            corrections[word] = corrected
                            corrected_words.append(corrected)
                            logger.info(f"Corrected: '{word}' -> '{corrected}' (distance={distance}, confidence={confidence:.2f})")
                        else:
                            # Keep original word if confidence is not high enough
                            corrected_words.append(word)
                            logger.debug(f"Low confidence: '{word}' -> '{corrected}' (confidence={confidence:.2f} < 0.9)")
                    else:
                        # Don't correct if distance > 1 (might not be a typo, could be domain term)
                        corrected_words.append(word)
                        logger.debug(f"Distance too high: '{word}' -> '{corrected}' (distance={distance}), keeping original")
                else:
                    corrected_words.append(word)
            except Exception as e:
                logger.debug(f"Spell check failed for '{word}': {e}")
                corrected_words.append(word)
        
        corrected_query = " ".join(corrected_words)
        
        if corrections:
            logger.info(f"Query spelling corrected: '{query}' -> '{corrected_query}' (corrections: {corrections})")
        
        return corrected_query, corrections
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return SpellingCorrectionEngine._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
