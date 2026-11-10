"""Conservative WordNet synonym expansion.

Only the primary senses of short queries are used, at most ``max_synonyms_per_word`` per word, and the original
query is kept intact: synonyms are appended, so a retrieval model that ignores them still sees the user's words.

Expansion is skipped - not failed - when WordNet is unavailable. The old engine carried two dead filters (a
similarity test that always compared a synset with itself, and a frequency floor that dropped valid synonyms), and
every exception was swallowed with a bare ``except``, which hid a broken corpus until it reached the service layer.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class QueryExpansionEngine:
    def __init__(
        self,
        max_synonyms_per_word: int = 2,
        max_synsets: int = 2,
        min_term_length: int = 4,
        wordnet_reader=None,
        stop_words: Optional[Set[str]] = None,
        max_query_words: int = 4,
        min_synonym_count: int = 2,
    ):
        self.max_synonyms_per_word = max_synonyms_per_word
        self.max_synsets = max_synsets
        self.min_term_length = min_term_length
        self.max_query_words = max_query_words
        self.min_synonym_count = min_synonym_count
        self.wordnet = wordnet_reader
        self.stop_words = set(stop_words) if stop_words is not None else set()

    @property
    def available(self) -> bool:
        return self.wordnet is not None

    def expand(self, query: str) -> Tuple[str, Dict[str, List[str]]]:
        """Return the expanded query and the synonyms added for each original word."""
        if not query or not query.strip() or not self.available:
            return query, {}

        words = query.split()
        if len(words) > self.max_query_words:
            logger.debug("Query has %d words, skipping expansion", len(words))
            return query, {}

        known = {word.lower() for word in words}
        expansions: Dict[str, List[str]] = {}
        added: List[str] = []

        for word in dict.fromkeys(words):  # unique, order preserved
            if not self._is_expandable(word.lower()):
                continue
            synonyms = self._synonyms(word.lower(), known)
            if not synonyms:
                continue
            expansions[word] = synonyms
            known.update(synonyms)
            added.extend(synonyms)
            logger.info("Expanded %r with %s", word, synonyms)

        if not added:
            return query, {}
        return f"{query} {' '.join(added)}", expansions

    # -- internals ---------------------------------------------------------
    def _is_expandable(self, word: str) -> bool:
        return word.isalpha() and len(word) >= self.min_term_length and word not in self.stop_words

    def _synonyms(self, word: str, existing: Set[str]) -> List[str]:
        try:
            synsets = self.wordnet.synsets(word)[: self.max_synsets]
        except Exception as exc:  # noqa: BLE001 - an unusable corpus only costs this word
            logger.warning("WordNet lookup failed for %r: %s", word, exc)
            return []

        ranked: List[Tuple[int, str]] = []
        for synset in synsets:
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if not self._is_valid_synonym(name, word, existing):
                    continue
                count = int(lemma.count())
                if count < self.min_synonym_count:
                    # No corpus evidence for the sense ("bound" for boundary, "bed" for layer): appending it only
                    # adds noise to a query that retrieval will run verbatim.
                    continue
                # Most frequent lemma first; the name keeps the order stable for equal counts.
                ranked.append((-count, name))

        selected: List[str] = []
        for _, name in sorted(ranked):
            if name not in selected:
                selected.append(name)
            if len(selected) >= self.max_synonyms_per_word:
                break
        return selected

    def _is_valid_synonym(self, synonym: str, original: str, existing: Set[str]) -> bool:
        return (
            synonym != original
            and synonym not in existing
            and synonym not in self.stop_words
            and synonym.isalpha()
            and len(synonym) >= 3
            and " " not in synonym
        )
