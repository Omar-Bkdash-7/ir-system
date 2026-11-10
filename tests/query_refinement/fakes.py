"""Deterministic stand-ins for the dictionary, WordNet, the embedding model and the NLTK corpora.

The refinement stages only need four collaborators, so the tests inject them instead of downloading corpora or
loading a sentence-transformer: the suite stays fast and never depends on the network.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np

from services.query_refinement.corpora import CorpusStatus
from services.query_refinement.engines import (
    QueryExpansionEngine,
    SemanticRefinementEngine,
    SpellingCorrectionEngine,
)
from services.query_refinement.services.refinement_manager import QueryRefinementService
from services.query_refinement.vocabulary import DatasetVocabulary

STOP_WORDS = {"a", "an", "the", "is", "are", "of", "in", "on", "with", "and", "to", "for", "how", "what"}


class FakeSpellChecker:
    """pyspellchecker look-alike: knows ``vocabulary``, suggests from ``corrections`` and exposes word frequencies."""

    def __init__(
        self,
        corrections: Optional[Dict[str, str]] = None,
        vocabulary: Iterable[str] = (),
        frequencies: Optional[Dict[str, int]] = None,
    ):
        self.corrections = dict(corrections or {})
        self.vocabulary = set(vocabulary) | set(self.corrections.values())
        self.word_frequency = Counter({word: 100_000 for word in self.vocabulary})
        for word, count in (frequencies or {}).items():
            self.word_frequency[word] = count

    def __contains__(self, word: str) -> bool:
        return word in self.vocabulary

    def correction(self, word: str) -> Optional[str]:
        return self.corrections.get(word)


class _Lemma:
    def __init__(self, name: str, count: int):
        self._name = name
        self._count = count

    def name(self) -> str:
        return self._name

    def count(self) -> int:
        return self._count


class _Synset:
    def __init__(self, lemmas: List[_Lemma]):
        self._lemmas = lemmas

    def lemmas(self) -> List[_Lemma]:
        return list(self._lemmas)


class FakeWordNet:
    """WordNet look-alike.

    ``{word: [["synonym", ...], ...]}`` declares synsets whose lemma counts decrease (100, 99, ...), so earlier
    synsets and earlier lemmas rank first. A synset written as ``{"synonym": count}`` uses those counts instead,
    which is how a lemma without corpus evidence (`count == 0`) is expressed.
    """

    def __init__(self, table: Dict[str, List]):
        self.table = table

    def synsets(self, word: str):
        entry = self.table.get(word)
        if isinstance(entry, Exception):
            raise entry
        synsets = []
        for synset_index, synset in enumerate(entry or []):
            if isinstance(synset, dict):
                lemmas = [_Lemma(name, count) for name, count in synset.items()]
            else:
                # Counts decrease across the whole entry, so "earlier synset / earlier lemma" ranks first.
                lemmas = [_Lemma(name, 100 - synset_index * 10 - index) for index, name in enumerate(synset)]
            synsets.append(_Synset(lemmas))
        return synsets


class FakeModel:
    """Bag-of-words hashing into 16 dimensions, normalised - related texts really do score higher."""

    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.calls += 1
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            for token in text.lower().split():
                vector[sum(map(ord, token)) % self.dimensions] += 1.0
            norm = float(np.linalg.norm(vector))
            vectors.append(vector / norm if norm else vector)
        return np.vstack(vectors)


class BrokenModel:
    def encode(self, *_args, **_kwargs):
        raise RuntimeError("model is broken")


class FakeCorpora:
    def __init__(self, wordnet_reader=None, stop_words: Optional[Iterable[str]] = None):
        self._wordnet = wordnet_reader
        self._stop_words = set(stop_words if stop_words is not None else STOP_WORDS)

    def wordnet(self):
        return self._wordnet

    def stopwords(self):
        return set(self._stop_words)

    def status(self):
        return {
            "wordnet": CorpusStatus("wordnet", self._wordnet is not None, "fake"),
            "stopwords": CorpusStatus("stopwords", True, "fake"),
        }


def make_service(
    *,
    wordnet=None,
    model=None,
    corrections: Optional[Dict[str, str]] = None,
    max_edit_distance: int = 2,
    confidence_threshold: float = 0.7,
    similarity_threshold: float = 0.1,
    max_drift: float = 0.25,
    max_query_words: int = 4,
    min_synonym_count: int = 1,
    vocabulary=None,
) -> QueryRefinementService:
    """A service whose stages are all fakes; thresholds are set by the test, not by config.yaml."""
    corpora = FakeCorpora(wordnet)
    service = QueryRefinementService(
        corpora=corpora,
        spelling_engine=SpellingCorrectionEngine(
            max_edit_distance=max_edit_distance,
            confidence_threshold=confidence_threshold,
            spell_checker=FakeSpellChecker(corrections),
            stop_words=STOP_WORDS,
        ),
        expansion_engine=QueryExpansionEngine(
            max_synonyms_per_word=2,
            max_synsets=2,
            min_term_length=4,
            wordnet_reader=wordnet,
            stop_words=STOP_WORDS,
            max_query_words=max_query_words,
            min_synonym_count=min_synonym_count,
        ),
        semantic_engine=SemanticRefinementEngine(model=model or FakeModel(), stop_words=STOP_WORDS),
        vocabulary=vocabulary or DatasetVocabulary(loader=lambda dataset: set()),
    )
    service.similarity_threshold = similarity_threshold
    service.max_drift = max_drift
    return service
