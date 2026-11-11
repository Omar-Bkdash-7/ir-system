"""HTTP layer of the gateway: routing, status mapping and pass-through.

Every downstream failure used to be swallowed by the generic handler and reported
as a 500, so a missing index (a plain 404 from the retrieval service) looked like a
gateway bug. The service object is faked here, so these tests describe the HTTP
contract only.
"""
import importlib.util
import logging
import os
import tempfile
import unittest

from common.http_client import DownstreamError

HAVE_FASTAPI = all(
    importlib.util.find_spec(module) is not None for module in ("fastapi", "httpx")
)


class FakeSearch:
    """Stands in for SearchService; `failure` makes the next call raise it."""

    def __init__(self):
        self.failure = None
        self.calls = []

    def _maybe_fail(self):
        if self.failure:
            raise self.failure

    def dataset_info(self):
        self._maybe_fail()
        return {"dataset_name": "toy", "doc_count": 3}

    def refine(self, query, history, enabled, session_id, is_evaluation, dataset):
        self._maybe_fail()
        self.calls.append({"query": query, "history": history, "session_id": session_id,
                           "is_evaluation": is_evaluation, "dataset": dataset})
        return {"refined_query": "refined query", "applied": True}

    def search_basic(self, req):
        self._maybe_fail()
        return {"results": [{"doc_id": "d1"}], "total": 1, "model": req.model}

    def search_hybrid_serial(self, req):
        self._maybe_fail()
        return {"results": [], "total": 0, "stage": "serial"}

    def search_hybrid_parallel(self, req):
        self._maybe_fail()
        return {"results": [], "total": 0, "stage": "parallel"}

    def close(self):
        pass


class FakeRag:
    def __init__(self):
        self.failure = None

    def _maybe_fail(self):
        if self.failure:
            raise self.failure

    async def generate(self, payload):
        self._maybe_fail()
        return {"answer": "hello", "echo": payload}

    async def health(self):
        self._maybe_fail()
        return {"status": "ok"}

    async def models(self):
        self._maybe_fail()
        return {"models": ["local"]}

    async def local_model(self):
        self._maybe_fail()
        return {"status": "absent"}

    async def download_local_model(self):
        self._maybe_fail()
        return {"started": True}


@unittest.skipUnless(HAVE_FASTAPI, "FastAPI test client (fastapi + httpx) is not installed")
class GatewayHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        previous = os.getcwd()
        # importing main calls setup_logging(), which writes logs/api_gateway.log
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
            os.chdir(scratch)
            try:
                from services.api_gateway.main import app
            finally:
                os.chdir(previous)

        cls.app = app
        cls.client = TestClient(app, raise_server_exceptions=False)
        logging.disable(logging.CRITICAL)
        cls.addClassCleanup(logging.disable, logging.NOTSET)

    def setUp(self):
        self.search = FakeSearch()
        self.rag = FakeRag()
        self.app.state.search = self.search
        self.app.state.rag = self.rag

    # --------------------------------------------------------------------------------------------- happy paths
    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "service": "api_gateway"})

    def test_dataset_info_is_served_from_the_service(self):
        self.assertEqual(self.client.get("/datasets/info").json()["dataset_name"], "toy")

    def test_search_returns_the_retrieval_answer(self):
        response = self.client.post("/search", json={"query": "heat transfer", "model": "bm25", "dataset": "toy"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 1)
        self.assertEqual(response.json()["model"], "bm25")

    def test_the_hybrid_routes_reach_their_own_service_methods(self):
        serial = self.client.post("/search/hybrid/serial",
                                  json={"query": "heat", "first_model": "bm25", "second_model": "tfidf"})
        parallel = self.client.post("/search/hybrid/parallel",
                                    json={"query": "heat", "models": ["bm25", "tfidf"]})

        self.assertEqual(serial.json()["stage"], "serial")
        self.assertEqual(parallel.json()["stage"], "parallel")

    def test_refinement_parameters_are_forwarded(self):
        response = self.client.post("/query/refine", json={
            "query": "informtion retrival", "history": ["a"], "session_id": "s1", "dataset": "toy",
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["refined_query"], "refined query")
        self.assertEqual(self.search.calls[0]["session_id"], "s1")
        self.assertEqual(self.search.calls[0]["dataset"], "toy")

    def test_rag_routes_pass_through(self):
        self.assertEqual(self.client.get("/rag/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/rag/models").json(), {"models": ["local"]})
        self.assertEqual(self.client.get("/rag/local-model").json(), {"status": "absent"})
        self.assertEqual(self.client.post("/rag/local-model/download").json(), {"started": True})

        generated = self.client.post("/rag/generate", json={"query": "what is IR?"})
        self.assertEqual(generated.json()["answer"], "hello")
        self.assertEqual(generated.json()["echo"], {"query": "what is IR?"})

    # --------------------------------------------------------------------------------------------- failures
    def test_a_blank_query_is_a_400_for_refinement(self):
        self.assertEqual(self.client.post("/query/refine", json={"query": ""}).status_code, 400)
        self.assertEqual(self.search.calls, [])

    def test_a_query_the_service_rejects_is_a_400_on_search_too(self):
        # Validation lives in the service (one place), so the HTTP layer maps its ValueError to a 400.
        self.search.failure = ValueError("Query cannot be empty")

        self.assertEqual(self.client.post("/search", json={"query": "   "}).status_code, 400)

    def test_a_missing_index_downstream_is_a_404_not_a_500(self):
        self.search.failure = DownstreamError(
            "retrieval", "TF-IDF index not found at data/indices/toy", status_code=404
        )

        response = self.client.post("/search", json={"query": "heat transfer", "dataset": "toy"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "TF-IDF index not found at data/indices/toy")

    def test_a_rejected_downstream_request_keeps_its_status(self):
        self.search.failure = DownstreamError(
            "retrieval", "Model must be one of ['tfidf', 'embedding', 'bm25']", status_code=422
        )

        response = self.client.post("/search/hybrid/parallel",
                                    json={"query": "heat", "models": ["bm25", "nope"]})

        self.assertEqual(response.status_code, 422)
        self.assertIn("Model must be one of", response.json()["detail"])

    def test_an_unreachable_service_is_a_502(self):
        self.search.failure = DownstreamError("retrieval", "unreachable after 2 attempts: Connection refused")

        self.assertEqual(self.client.post("/search", json={"query": "heat"}).status_code, 502)

    def test_an_unreachable_rag_service_is_a_502(self):
        self.rag.failure = DownstreamError("rag", "ConnectError: all connection attempts failed")

        self.assertEqual(self.client.get("/rag/health").status_code, 502)

    def test_an_unexpected_failure_is_still_a_500(self):
        for path, payload in (
            ("/search", {"query": "heat"}),
            ("/search/hybrid/serial", {"query": "heat", "first_model": "bm25", "second_model": "tfidf"}),
            ("/search/hybrid/parallel", {"query": "heat", "models": ["bm25", "tfidf"]}),
        ):
            with self.subTest(path=path):
                self.search.failure = RuntimeError("boom")
                self.assertEqual(self.client.post(path, json=payload).status_code, 500)


if __name__ == "__main__":
    unittest.main()
