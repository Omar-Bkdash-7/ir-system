"""Old retrievers vs the new indexes, on the REAL index files of the project.

For every dataset that has both a BM25 and a TF-IDF index under data/indices (run from the project root) it checks, for many
queries, many values of k and several BM25 parameter sets, that the new code returns the same documents with the same
scores as the code it replaced (tests/retrieval/legacy), and that hybrid-serial re-scoring is identical.

Preprocessing is switched off on the legacy side (tokens in = tokens out), because this compares the SCORING, not spaCy.
Documents that tie on score may come back in a different order (the new order is deterministic: corpus order), so the
check is: the k scores are identical, and every returned document has the same score in both implementations.

    RETRIEVAL_TEST_DATASETS=cranfield,vaswani   datasets to compare (default: all found, at most 3)
    RETRIEVAL_TEST_QUERIES=150                  number of random queries per dataset
"""
import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from services.retrieval.core import indexes as I
from services.retrieval.core.compact import CompactCache
from services.retrieval.core.query import QueryContext
from services.retrieval.core.store import IndexStore

_ORIGINAL_PREPROCESS = None
_STUB_NAME = "services.preprocessing.preprocessing_service"
legacy_bm25 = legacy_tfidf = None


def setUpModule():
    global _ORIGINAL_PREPROCESS, legacy_bm25, legacy_tfidf
    _ORIGINAL_PREPROCESS = sys.modules.get(_STUB_NAME)
    stub = types.ModuleType(_STUB_NAME)
    stub.preprocess = lambda text, mode=None, config=None: text.split()
    sys.modules[_STUB_NAME] = stub
    legacy_bm25 = importlib.import_module("tests.retrieval.legacy.bm25_retriever")
    legacy_tfidf = importlib.import_module("tests.retrieval.legacy.tfidf_retriever")


def tearDownModule():
    if _ORIGINAL_PREPROCESS is not None:
        sys.modules[_STUB_NAME] = _ORIGINAL_PREPROCESS
    else:
        sys.modules.pop(_STUB_NAME, None)


def _restore_idf(vectorizer):
    """A vectorizer pickled by an older scikit-learn keeps its weights in a diagonal matrix: expose them as idf_."""
    transformer = vectorizer._tfidf
    try:
        transformer.idf_
    except AttributeError:
        transformer.idf_ = np.asarray(transformer._idf_diag.sum(axis=0)).ravel()


def _datasets():
    wanted = os.getenv("RETRIEVAL_TEST_DATASETS")
    if wanted:
        return [name.strip() for name in wanted.split(",") if name.strip()]
    store = IndexStore(cache=CompactCache(None))
    return [d for d in store.known_datasets() if store.is_built(d, "bm25") and store.is_built(d, "tfidf")][:3]


class DifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = _datasets()
        if not cls.datasets:
            raise unittest.SkipTest("no dataset with BM25 and TF-IDF indexes under data/indices (run from the project root)")
        cls.cache = CompactCache(Path(tempfile.mkdtemp(prefix="retrieval-diff-")))
        cls.cases = {}
        import warnings
        warnings.simplefilter("ignore")
        for dataset in cls.datasets:
            old_bm25 = legacy_bm25.BM25Retriever(dataset)
            old_bm25.load()
            old_tfidf = legacy_tfidf.TFIDFRetriever(dataset)
            old_tfidf.load()
            _restore_idf(old_tfidf.vectorizer)
            new_bm25 = I.load_index("bm25", dataset, embedding_mmap=True, cache=cls.cache)
            new_tfidf = I.load_index("tfidf", dataset, embedding_mmap=True, cache=cls.cache)
            cls.cases[dataset] = (old_bm25, old_tfidf, new_bm25, new_tfidf, cls._queries(new_bm25))

    @staticmethod
    def _queries(index):
        rng = np.random.default_rng(2024)
        vocabulary = list(index._terms)
        frequency = np.diff(index._offsets).astype(float)
        probability = frequency / frequency.sum()
        count = int(os.getenv("RETRIEVAL_TEST_QUERIES", "150"))
        queries = []
        for number in range(count):
            size = int(rng.integers(1, 9))
            if number % 3 == 0:
                tokens = [vocabulary[i] for i in rng.choice(len(vocabulary), size=size, p=probability)]  # common words
            elif number % 3 == 1:
                tokens = [vocabulary[i] for i in rng.choice(len(vocabulary), size=size)]                  # rarer words
            else:
                tokens = [vocabulary[i] for i in rng.choice(len(vocabulary), size=3, p=probability)]
                tokens += tokens[:2] + ["zzzunknownword"]                                                   # repeats + unknown
            queries.append(tokens)
        queries.append([vocabulary[0]])
        return queries

    @staticmethod
    def _context(tokens):
        return QueryContext(text="", tokens=tokens, encoder=None, preprocess=lambda text: text.split())

    def _compare(self, label, old_search, new_search, queries, n_docs, ks=(1, 10, 100, 1000), tol=1e-9):
        for tokens in queries:
            full = old_search(tokens, n_docs)
            score_of = {item["doc_id"]: item["score"] for item in full}
            for k in ks:
                expected = full[:k]
                ids, scores = new_search(tokens, k)
                self.assertEqual(len(ids), len(expected), (label, tokens, k))
                if not ids:
                    continue
                old_scores = np.array([item["score"] for item in expected])
                np.testing.assert_allclose(scores, old_scores, rtol=tol, atol=1e-12, err_msg=f"{label} {tokens} k={k}")
                for doc_id, score in zip(ids, scores):
                    self.assertAlmostEqual(score, score_of[doc_id], delta=tol * max(1.0, abs(score)), msg=f"{label} {tokens} {doc_id}")

    def test_bm25_matches_the_legacy_loop(self):
        for dataset, (old, _, new, _, queries) in self.cases.items():
            for k1, b in ((1.5, 0.75), (1.2, 0.4), (2.5, 1.0), (0.5, 0.0)):
                def new_search(tokens, k, k1=k1, b=b):
                    hits = new.search(new.prepare(self._context(tokens)), k, I.SearchOptions(k1, b))
                    return new.ids_for(hits.rows), hits.scores.tolist()
                self._compare(f"{dataset} bm25 k1={k1} b={b}", lambda t, k, k1=k1, b=b: old.retrieve(t, top_k=k, k1=k1, b=b),
                              new_search, queries, new.n_docs)

    def test_tfidf_matches_cosine_similarity(self):
        for dataset, (_, old, _, new, queries) in self.cases.items():
            def new_search(tokens, k):
                hits = new.search(new.prepare(self._context(tokens)), k, I.SearchOptions())
                return new.ids_for(hits.rows), hits.scores.tolist()
            self._compare(f"{dataset} tfidf", lambda t, k: old.retrieve(t, top_k=k), new_search, queries, new.n_docs)

    def test_hybrid_serial_rescoring_matches_score_documents(self):
        for dataset, (old_bm25, old_tfidf, new_bm25, new_tfidf, queries) in self.cases.items():
            pairs = {"bm25": (old_bm25, new_bm25), "tfidf": (old_tfidf, new_tfidf)}
            for first, (old_first, _) in pairs.items():
                for second, (old_second, new_second) in pairs.items():
                    for tokens in queries[:60]:
                        candidates = [item["doc_id"] for item in old_first.retrieve(tokens, top_k=200)]
                        if not candidates:
                            continue
                        expected = old_second.score_documents(tokens, candidates)
                        scores = new_second.score(new_second.prepare(self._context(tokens)), new_second.rows_for(candidates), I.SearchOptions())
                        np.testing.assert_allclose(scores, [expected[c] for c in candidates], rtol=1e-9, atol=1e-12,
                                                   err_msg=f"{dataset} {first}->{second} {tokens}")


if __name__ == "__main__":
    unittest.main()
