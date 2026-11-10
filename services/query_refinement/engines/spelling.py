"""Dictionary-based spelling correction that keeps the shape of the query.

The old implementation lower-cased the whole query (``NASA missions`` came back as ``nasa missions``) and ignored
the thresholds it was given: ``__init__`` accepted ``max_edit_distance``/``confidence_threshold`` and then
hard-coded 2 and 0.7, so configuration could not change the behaviour. Tokens are now corrected in place, one at a
time, leaving case, punctuation and spacing untouched.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Set, Tuple

from spellchecker import SpellChecker

logger = logging.getLogger(__name__)

# Letters only: digits, underscores and punctuation are never "misspelled words" and must survive unchanged.
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Above this many words the dictionary check is repeated per word from a cache anyway, but the set should not grow
# without bound in a long-running service.
_CACHE_LIMIT = 4096


class SpellingCorrectionEngine:
    def __init__(
        self,
        max_edit_distance: int = 2,
        confidence_threshold: float = 0.7,
        spell_checker=None,
        stop_words: Optional[Set[str]] = None,
        min_correction_frequency: int = 1000,
    ):
        self.spell_checker = spell_checker or SpellChecker()
        self.max_edit_distance = max_edit_distance
        self.confidence_threshold = confidence_threshold
        # Rare words are almost never the right replacement for an unknown word: the dictionary turned "aeroelastic"
        # into "ceroplastic" (frequency 50) and "forebody" into "forebode" (50), while real fixes (boundary 3252,
        # question 213832) are common words.
        self.min_correction_frequency = min_correction_frequency
        self._frequency = getattr(self.spell_checker, "word_frequency", {})
        self.stop_words = set(stop_words) if stop_words else self._load_stopwords()
        self.protected_terms = self._load_protected_terms()
        self._suggestions: Dict[str, Optional[str]] = {}

    def add_domain_terms(self, terms: Set[str]) -> None:
        self.protected_terms.update(term.lower() for term in terms)
        self._suggestions.clear()

    def correct(self, query: str, known_words: Optional[Set[str]] = None) -> Tuple[str, Dict[str, str]]:
        """Return the query with misspelled words replaced in place, plus the corrections that were applied.

        ``known_words`` is the vocabulary of the dataset being searched: a word the corpus uses is correct by
        definition, whatever the dictionary thinks of it.
        """
        if not query or not query.strip():
            return query, {}

        known = {word.lower() for word in known_words} if known_words else set()
        corrections: Dict[str, str] = {}

        def replace(match: "re.Match[str]") -> str:
            word = match.group(0)
            if _is_proper_noun(word, query, match.start()):
                # "Quora" was being rewritten to "Quota" and "NASA" to "Nasa": a capitalised word is a name, and
                # the dictionary is the wrong authority on names.
                return word
            fixed = self._correct_word(word, known)
            if fixed is None:
                return word
            corrections[word] = fixed
            return fixed

        corrected = _WORD.sub(replace, query)
        if corrections:
            logger.info("Spelling corrected %r -> %r (%s)", query, corrected, corrections)
        return corrected, corrections

    # -- internals ---------------------------------------------------------
    def _correct_word(self, word: str, known: Set[str]) -> Optional[str]:
        lower = word.lower()
        if len(lower) <= 3 or lower in self.stop_words or lower in self.protected_terms:
            return None
        if lower in known:
            # "pitot", "planform" and "mach" are ordinary words in the corpus and misspellings nowhere else.
            return None

        candidate = self._suggestion(lower)
        if candidate is None or candidate == lower:
            return None

        if _is_spelling_variant(lower, candidate):
            # behaviour -> behavior is a regional variant, not a typo: rewriting it breaks matching against a corpus
            # that spells the word the British way.
            return None

        if self._frequency_count(candidate) < self.min_correction_frequency:
            logger.debug("Rare candidate for %r: %r", word, candidate)
            return None

        distance = levenshtein_distance(lower, candidate)
        if distance > self.max_edit_distance:
            logger.debug("Distance too high: %r -> %r (%d)", word, candidate, distance)
            return None

        confidence = 1.0 - distance / max(len(lower), len(candidate))
        if confidence < self.confidence_threshold or character_similarity(lower, candidate) < 0.6:
            logger.debug("Low confidence: %r -> %r (%.2f)", word, candidate, confidence)
            return None

        logger.info("Corrected %r -> %r (distance=%d, confidence=%.2f)", word, candidate, distance, confidence)
        return _match_case(word, candidate)

    def _suggestion(self, lower: str) -> Optional[str]:
        if lower in self._suggestions:
            return self._suggestions[lower]

        candidate: Optional[str] = None
        try:
            # The dictionary is checked first: a known word is never "corrected" to a rarer neighbour.
            if lower not in self.spell_checker:
                candidate = self.spell_checker.correction(lower)
        except Exception as exc:  # noqa: BLE001 - a dictionary failure must not fail the request
            logger.debug("Spell check failed for %r: %s", lower, exc)

        if len(self._suggestions) >= _CACHE_LIMIT:
            self._suggestions.clear()
        self._suggestions[lower] = candidate
        return candidate

    def _frequency_count(self, word: str) -> int:
        try:
            return int(self._frequency[word])
        except (KeyError, TypeError, ValueError):
            return 0

    @staticmethod
    def _load_stopwords() -> Set[str]:
        try:
            from nltk.corpus import stopwords as nltk_stopwords
            return set(nltk_stopwords.words("english"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stopword corpus unavailable (%s), using the built-in list", exc)
            from services.query_refinement.corpora import FALLBACK_STOPWORDS
            return set(FALLBACK_STOPWORDS)

    @staticmethod
    def _load_protected_terms() -> Set[str]:
        # Technical vocabulary the dictionary does not know; correcting it is always wrong for this system.
        return {
            "api", "http", "https", "json", "xml", "html", "css", "sql", "nosql", "mongodb", "redis", "kafka",
            "nginx", "docker", "kubernetes", "oauth", "jwt", "cors", "crud", "rest", "graphql", "grpc", "devops",
            "mlops", "nlp", "bert", "gpu", "cpu", "ram", "ssd", "raid", "vpc", "cdn", "dns", "ssh", "ftp", "smtp",
            "tcp", "udp", "lan", "wan", "aws", "azure", "gcp", "github", "gitlab", "bitbucket", "tfidf", "bm25",
            "ndcg", "map", "trec", "ir", "rag", "llm", "embedding", "tokenizer", "stemming", "lemmatization",
        }


# British -> American spelling pairs. A correction that only crosses one of these is a regional variant, and
# rewriting it changes the token the index stored.
_VARIANT_PATTERNS = (
    (re.compile(r"our(s?)$"), r"or\1"),
    (re.compile(r"ys(e|es|ed|ing)$"), r"yz\1"),
    (re.compile(r"is(e|es|ed|ing|ation)$"), r"iz\1"),
    (re.compile(r"ogue$"), "og"),
    (re.compile(r"ll(ed|ing|er)$"), r"l\1"),
    (re.compile(r"re$"), "er"),
    (re.compile(r"(ae|oe)"), "e"),
)


def _normalise_variant(word: str) -> str:
    for pattern, replacement in _VARIANT_PATTERNS:
        word = pattern.sub(replacement, word)
    return word


def _is_spelling_variant(original: str, candidate: str) -> bool:
    return original != candidate and _normalise_variant(original) == _normalise_variant(candidate)


def _is_proper_noun(word: str, query: str, start: int) -> bool:
    """Acronyms and capitalised words after the first are treated as names and left untouched."""
    if word.isupper() and len(word) > 1:
        return True
    return word[:1].isupper() and bool(query[:start].strip())


def _match_case(original: str, correction: str) -> str:
    """Keep the capitalisation the user typed: NASA stays NASA, Boundary stays Boundary."""
    if original.isupper():
        return correction.upper()
    if original[:1].isupper() and original[1:].islower():
        return correction.capitalize()
    return correction


def character_similarity(first: str, second: str) -> float:
    """Jaccard similarity over character sets - cheap guard against unrelated substitutions."""
    if not first or not second:
        return 0.0
    first_chars, second_chars = set(first.lower()), set(second.lower())
    union = first_chars | second_chars
    return len(first_chars & second_chars) / len(union) if union else 0.0


def levenshtein_distance(first: str, second: str) -> int:
    """Edit distance between two strings (insertions, deletions, substitutions)."""
    if len(first) < len(second):
        return levenshtein_distance(second, first)
    if not second:
        return len(first)

    previous_row = range(len(second) + 1)
    for i, first_char in enumerate(first):
        current_row = [i + 1]
        for j, second_char in enumerate(second):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (first_char != second_char)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
