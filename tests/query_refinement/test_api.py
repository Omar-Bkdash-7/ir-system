"""The HTTP surface: contract, status codes, and the broken-corpus regression."""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from services.query_refinement import main as main_module
from services.query_refinement.services import refinement_manager
from services.query_refinement.vocabulary import DatasetVocabulary
from tests.query_refinement.fakes import FakeWordNet, make_service

WORDNET = FakeWordNet({"boundary": [["boundary", "edge"]]})
CONTRACT_FIELDS = (
    "original_query", "refined_query", "corrections", "expansions", "history_boosts", "history_terms",
    "history_similarity", "confidence", "changes_applied", "semantic_drift", "applied", "evaluation_mode",
    "drift_rejected",
)


class ApiTestCase(unittest.TestCase):
    def client(self, service) -> TestClient:
        patcher = mock.patch.object(refinement_manager, "_service", service)
        patcher.start()
        self.addCleanup(patcher.stop)
        return TestClient(main_module.app)


class HealthTest(ApiTestCase):
    def test_health_reports_the_usable_stages(self):
        with self.client(make_service(wordnet=WORDNET)) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["features"]["synonym_expansion"])
        self.assertTrue(body["corpora"]["wordnet"]["available"])

    def test_health_stays_200_but_degrades_without_wordnet(self):
        with self.client(make_service(wordnet=None)) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertFalse(response.json()["features"]["synonym_expansion"])


class RefineEndpointTest(ApiTestCase):
    def test_the_response_carries_the_whole_contract(self):
        service = make_service(wordnet=WORDNET, corrections={"recieve": "receive"})
        with self.client(service) as client:
            response = client.post("/refine", json={"query": "recieve boundary", "history": []})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for field in CONTRACT_FIELDS:
            self.assertIn(field, body)
        self.assertTrue(body["refined_query"].startswith("receive boundary"))
        self.assertEqual(body["corrections"], {"recieve": "receive"})

    def test_a_blank_query_is_a_400_with_a_message(self):
        with self.client(make_service()) as client:
            response = client.post("/refine", json={"query": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_a_missing_query_is_a_422(self):
        with self.client(make_service()) as client:
            self.assertEqual(client.post("/refine", json={}).status_code, 422)

    def test_a_broken_corpus_no_longer_turns_refine_into_a_500(self):
        # Regression: the NLTK setup raised inside the singleton's constructor, so every request answered 500.
        service = make_service(wordnet=None, corrections={"recieve": "receive"})
        with self.client(service) as client:
            response = client.post("/refine", json={"query": "recieve data"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["refined_query"], "receive data")

    def test_evaluation_mode_and_session_id_are_honoured(self):
        service = make_service(wordnet=WORDNET)
        with self.client(service) as client:
            body = client.post(
                "/refine", json={"query": "boundary", "is_evaluation": True, "session_id": "session-1"}
            ).json()
        self.assertTrue(body["evaluation_mode"])
        self.assertEqual(body["expansions"], {})

    def test_the_dataset_is_forwarded_so_corpus_terms_stay_intact(self):
        service = make_service(
            corrections={"pitot": "pilot"},
            vocabulary=DatasetVocabulary(loader=lambda dataset: {"pitot"}),
        )
        with self.client(service) as client:
            body = client.post("/refine", json={"query": "pitot tube", "dataset": "cranfield"}).json()
        self.assertEqual(body["refined_query"], "pitot tube")
        self.assertEqual(body["corrections"], {})

    def test_disabling_refinement_is_forwarded_to_the_service(self):
        with self.client(make_service(corrections={"recieve": "receive"})) as client:
            body = client.post("/refine", json={"query": "recieve data", "enabled": False}).json()
        self.assertEqual(body["refined_query"], "recieve data")
        self.assertFalse(body["applied"])


if __name__ == "__main__":
    unittest.main()
