"""The pipeline as a whole: what each stage changes, and what happens when one is unavailable."""
import unittest

from services.query_refinement.engines import SemanticRefinementEngine
from services.query_refinement.vocabulary import DatasetVocabulary
from tests.query_refinement.fakes import STOP_WORDS, FakeWordNet, make_service

WORDNET = FakeWordNet({"boundary": [["boundary", "edge"]]})


class ValidationTest(unittest.TestCase):
    def test_an_empty_query_is_a_value_error(self):
        for query in ("", "   "):
            with self.subTest(query=query), self.assertRaises(ValueError):
                make_service().refine(query)

    def test_a_non_string_query_is_a_value_error(self):
        with self.assertRaises(ValueError):
            make_service().refine(None)


class DisabledTest(unittest.TestCase):
    def test_disabled_refinement_returns_the_query_untouched(self):
        result = make_service(corrections={"recieve": "receive"}).refine("recieve data", enabled=False)
        self.assertEqual(result["refined_query"], "recieve data")
        self.assertFalse(result["applied"])
        self.assertEqual(result["changes_applied"], [])


class EvaluationModeTest(unittest.TestCase):
    def test_only_spelling_is_applied(self):
        result = make_service(wordnet=WORDNET, corrections={"recieve": "receive"}).refine(
            "recieve boundary", is_evaluation=True
        )
        self.assertEqual(result["refined_query"], "receive boundary")
        self.assertEqual(result["expansions"], {})
        self.assertTrue(result["evaluation_mode"])
        self.assertEqual(result["changes_applied"], ["spelling"])

    def test_without_corrections_nothing_is_applied(self):
        result = make_service(wordnet=WORDNET).refine("boundary layer", is_evaluation=True)
        self.assertEqual(result["refined_query"], "boundary layer")
        self.assertFalse(result["applied"])
        self.assertEqual(result["confidence"], 0.0)


class PipelineTest(unittest.TestCase):
    def test_spelling_and_expansion_both_apply(self):
        result = make_service(wordnet=WORDNET, corrections={"recieve": "receive"}).refine("recieve boundary")
        self.assertTrue(result["refined_query"].startswith("receive boundary"))
        self.assertIn("edge", result["refined_query"])
        self.assertEqual(result["changes_applied"], ["spelling", "expansion"])
        self.assertTrue(result["applied"])
        self.assertGreater(result["confidence"], 0.0)

    def test_a_long_query_is_not_expanded(self):
        result = make_service(wordnet=WORDNET).refine("what is the boundary layer control theory")
        self.assertEqual(result["refined_query"], "what is the boundary layer control theory")
        self.assertFalse(result["applied"])

    def test_nothing_to_change_reports_not_applied(self):
        result = make_service(wordnet=FakeWordNet({})).refine("boundary layer")
        self.assertEqual(result["refined_query"], "boundary layer")
        self.assertFalse(result["applied"])
        self.assertEqual(result["confidence"], 0.0)

    def test_history_terms_are_appended_and_boosted(self):
        service = make_service(max_drift=0.6)
        result = service.refine(
            "control theory",
            ["boundary layer control", "boundary layer analysis"],
        )
        self.assertEqual(sorted(result["history_terms"]), ["boundary", "layer"])
        self.assertEqual(result["history_boosts"], {"boundary": 0.15, "layer": 0.15})
        self.assertTrue(result["refined_query"].endswith("boundary layer"))
        self.assertIn("history", result["changes_applied"])
        self.assertGreater(result["history_similarity"], 0.0)

    def test_one_query_never_borrows_terms_from_a_previous_one(self):
        # Evaluation runs every query on the same session with no history; the service must stay stateless.
        service = make_service(max_drift=0.6)
        first = service.refine("boundary layer control", session_id="s1")
        second = service.refine("turbulence model", session_id="s1")
        self.assertTrue(first["applied"] or not first["history_terms"])
        self.assertEqual(second["history_terms"], [])
        self.assertEqual(second["refined_query"], "turbulence model")

    def test_excessive_drift_reverts_to_the_corrected_query(self):
        service = make_service(wordnet=FakeWordNet({"boundary": [["boundary", "cooking"]]}), max_drift=0.25)
        result = service.refine("boundary")
        self.assertTrue(result["drift_rejected"])
        self.assertEqual(result["refined_query"], "boundary")
        self.assertEqual(result["expansions"], {})
        self.assertFalse(result["applied"])

    def test_drift_rejection_keeps_a_spelling_correction(self):
        service = make_service(
            wordnet=FakeWordNet({"boundary": [["boundary", "cooking"]]}),
            corrections={"boudary": "boundary"},
            max_drift=0.25,
        )
        result = service.refine("boudary")
        self.assertTrue(result["drift_rejected"])
        self.assertEqual(result["refined_query"], "boundary")
        self.assertEqual(result["changes_applied"], ["spelling"])

    def test_a_missing_corpus_only_disables_expansion(self):
        result = make_service(wordnet=None, corrections={"recieve": "receive"}).refine("recieve data")
        self.assertEqual(result["refined_query"], "receive data")
        self.assertEqual(result["expansions"], {})
        self.assertTrue(result["applied"])

    def test_an_unavailable_model_only_disables_history_and_drift(self):
        service = make_service(wordnet=WORDNET, corrections={"recieve": "receive"})
        service.semantic_engine = SemanticRefinementEngine(
            model_loader=lambda **_: (_ for _ in ()).throw(RuntimeError("no weights")),
            stop_words=STOP_WORDS,
        )
        result = service.refine("recieve boundary", ["boundary layer control", "boundary layer analysis"])
        self.assertEqual(result["history_terms"], [])
        self.assertEqual(result["semantic_drift"], 0.0)
        self.assertTrue(result["refined_query"].startswith("receive boundary"))


class DatasetVocabularyTest(unittest.TestCase):
    def test_domain_terms_are_protected_when_the_dataset_is_known(self):
        service = make_service(
            corrections={"pitot": "pilot"},
            vocabulary=DatasetVocabulary(loader=lambda dataset: {"pitot"} if dataset == "cranfield" else set()),
        )
        protected = service.refine("pitot static", dataset="cranfield")
        unprotected = service.refine("pitot static", dataset="other-dataset")
        self.assertEqual(protected["refined_query"], "pitot static")
        self.assertEqual(protected["corrections"], {})
        self.assertEqual(unprotected["corrections"], {"pitot": "pilot"})


class StatusTest(unittest.TestCase):
    def test_feature_status_reflects_the_available_stages(self):
        healthy = make_service(wordnet=WORDNET).feature_status()
        self.assertTrue(healthy["features"]["spelling_correction"])
        self.assertTrue(healthy["features"]["synonym_expansion"])
        self.assertTrue(healthy["corpora"]["wordnet"]["available"])

        degraded = make_service(wordnet=None).feature_status()
        self.assertTrue(degraded["features"]["spelling_correction"])
        self.assertFalse(degraded["features"]["synonym_expansion"])


class ConfidenceTest(unittest.TestCase):
    def test_confidence_stays_within_bounds(self):
        service = make_service(wordnet=WORDNET, corrections={"recieve": "receive"})
        for query in ("recieve boundary", "boundary layer", "unrelated words here"):
            with self.subTest(query=query):
                confidence = service.refine(query)["confidence"]
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
