"""Cross-service contract test: the API gateway against real services.

Every other test in this repository fakes its neighbours, so a renamed field or a
changed status code between two services stayed invisible until someone clicked
through the UI. This file runs against the stack in `docker-compose.ci.yml`
(postgres + preprocessing + retrieval + api_gateway) and asserts only the contract
the frontend depends on.

Run it with:
    docker compose -f docker-compose.ci.yml up -d --wait postgres preprocessing retrieval api_gateway
    docker compose -f docker-compose.ci.yml run --rm seed
    docker compose -f docker-compose.ci.yml run --rm tests
"""
import os
import time
import unittest

import httpx

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
DATASET = "ci/demo"


def wait_for_gateway(seconds: int = 90) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if httpx.get(f"{GATEWAY_URL}/health", timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return False


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wait_for_gateway():
            raise unittest.SkipTest(f"gateway unreachable at {GATEWAY_URL}")
        cls.client = httpx.Client(base_url=GATEWAY_URL, timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    # ----------------------------------------------------------------------------------------------- health & data
    def test_health(self):
        body = self.client.get("/health").json()

        self.assertEqual((body["status"], body["service"]), ("ok", "api_gateway"))

    def test_dataset_info_is_read_from_the_metadata_the_indexer_writes(self):
        body = self.client.get("/datasets/info").json()

        self.assertEqual(body["dataset_name"], DATASET)
        self.assertEqual(body["doc_count"], 3)

    # ----------------------------------------------------------------------------------------------- search
    def test_search_returns_the_matching_document(self):
        response = self.client.post(
            "/search",
            json={"query": "information retrieval", "model": "tfidf", "dataset": DATASET, "top_k": 3},
        )

        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()["results"]
        self.assertTrue(results, "the seeded index must answer this query")
        self.assertEqual(results[0]["doc_id"], "d1")
        self.assertIn("information retrieval", results[0]["text"])
        self.assertGreater(results[0]["score"], 0)

    def test_hybrid_parallel_fuses_the_two_indices(self):
        response = self.client.post(
            "/search/hybrid/parallel",
            json={"query": "bm25 term frequencies", "models": ["tfidf", "bm25"], "dataset": DATASET, "top_k": 2},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreaterEqual(len(response.json()["results"]), 1)

    def test_the_gateway_rejects_an_empty_query_before_calling_anything(self):
        response = self.client.post("/search", json={"query": "   "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Query cannot be empty")

    def test_a_missing_index_keeps_the_status_retrieval_answered_with(self):
        """The gateway used to turn every downstream failure into a 500."""
        response = self.client.post(
            "/search", json={"query": "anything", "model": "tfidf", "dataset": "not/indexed", "top_k": 1}
        )

        self.assertEqual(response.status_code, 404, response.text)
        self.assertIn("index", response.json()["detail"].lower())

    # ----------------------------------------------------------------------------------------------- protections
    def test_the_gateway_serves_cors_headers(self):
        response = self.client.get("/health", headers={"Origin": "http://localhost:8501"})

        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:8501")


if __name__ == "__main__":
    unittest.main()
