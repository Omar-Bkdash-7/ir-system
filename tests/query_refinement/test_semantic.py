"""Similarity and drift: loaded lazily, never fatal, and neutral when the model is missing."""
import unittest

from services.query_refinement.engines import SemanticRefinementEngine
from tests.query_refinement.fakes import STOP_WORDS, BrokenModel, FakeModel


def engine(model=None, loader=None) -> SemanticRefinementEngine:
    return SemanticRefinementEngine(model=model, model_loader=loader, stop_words=STOP_WORDS)


class SimilarityTest(unittest.TestCase):
    def test_a_related_earlier_query_scores_higher_than_an_unrelated_one(self):
        engine_ = engine(FakeModel())
        related = engine_.compute_similarity("boundary layer control", ["boundary layer theory"])
        unrelated = engine_.compute_similarity("boundary layer control", ["cooking pasta recipes"])
        self.assertGreater(related, unrelated)
        self.assertGreater(related, 0.3)

    def test_similarity_is_zero_without_history(self):
        self.assertEqual(engine(FakeModel()).compute_similarity("anything", []), 0.0)

    def test_similarity_is_zero_when_the_model_cannot_be_loaded(self):
        engine_ = engine(loader=lambda **_: (_ for _ in ()).throw(RuntimeError("no weights")))
        self.assertEqual(engine_.compute_similarity("query", ["history"]), 0.0)
        self.assertFalse(engine_.available)

    def test_the_model_is_loaded_once_on_first_use(self):
        calls = []

        def loader(**kwargs):
            calls.append(kwargs)
            return FakeModel()

        engine_ = engine(loader=loader)
        self.assertFalse(engine_.model_loaded)
        self.assertEqual(calls, [])
        engine_.compute_similarity("query", ["history"])
        engine_.compute_similarity("other query", ["history"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(engine_.model_loaded)

    def test_an_encoding_failure_returns_no_vector_instead_of_raising(self):
        self.assertIsNone(engine(BrokenModel()).encode("query"))


class DriftTest(unittest.TestCase):
    def test_an_identical_query_has_no_drift(self):
        self.assertEqual(engine(FakeModel()).validate_semantic_drift("query", "query"), (True, 0.0))

    def test_a_meaning_changing_rewrite_is_rejected(self):
        acceptable, drift = engine(FakeModel()).validate_semantic_drift("boundary layer", "cooking pasta")
        self.assertFalse(acceptable)
        self.assertGreater(drift, 0.25)

    def test_a_close_rewrite_is_accepted(self):
        acceptable, drift = engine(FakeModel()).validate_semantic_drift("boundary layer", "boundary layer edge")
        self.assertTrue(acceptable)
        self.assertLess(drift, 0.25)

    def test_drift_is_accepted_when_embeddings_are_unavailable(self):
        engine_ = engine(loader=lambda **_: (_ for _ in ()).throw(RuntimeError("no weights")))
        self.assertEqual(engine_.validate_semantic_drift("a", "b"), (True, 0.0))


class HistoryTermsTest(unittest.TestCase):
    HISTORY = ["boundary layer control", "boundary layer analysis"]

    def test_terms_repeated_in_the_session_are_extracted(self):
        terms = engine(FakeModel()).extract_relevant_history_terms("turbulence model", self.HISTORY, 0.8)
        self.assertEqual(sorted(terms), ["boundary", "layer"])

    def test_terms_below_the_similarity_threshold_are_ignored(self):
        self.assertEqual(engine(FakeModel()).extract_relevant_history_terms("q", self.HISTORY, 0.05), [])

    def test_a_term_seen_once_is_not_used(self):
        terms = engine(FakeModel()).extract_relevant_history_terms("turbulence model", ["boundary control"], 0.8)
        self.assertEqual(terms, [])

    def test_query_words_are_not_repeated(self):
        terms = engine(FakeModel()).extract_relevant_history_terms("boundary layer", self.HISTORY, 0.8)
        self.assertEqual(terms, [])

    def test_at_most_max_terms_are_returned(self):
        history = ["alpha beta gamma delta", "alpha beta gamma delta"]
        terms = engine(FakeModel()).extract_relevant_history_terms("unrelated", history, 0.8, max_terms=2)
        self.assertEqual(len(terms), 2)


if __name__ == "__main__":
    unittest.main()
