"""HTTP contract of the retrieval service: routes, status codes and JSON shapes (needs fastapi + httpx + pydantic)."""
import unittest

from services.retrieval.config import RetrievalSettings
from services.retrieval.core.compact import CompactCache
from services.retrieval.core.service import RetrievalService
from services.retrieval.core.store import IndexStore
from tests.retrieval.helpers import (TOY_DOCS, FakeEncoder, FakeTexts, project_dir, tokens_of, write_bm25, write_embeddings,
                                     write_tfidf)

try:
    from fastapi.testclient import TestClient

    from services.retrieval import main
    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover
    HAVE_FASTAPI = False


@unittest.skipUnless(HAVE_FASTAPI, "fastapi / httpx / pydantic are not installed")
class HttpTests(unittest.TestCase):
    def setUp(self):
        self._ctx = project_dir()
        root = self._ctx.__enter__()
        self.addCleanup(self._ctx.__exit__, None, None, None)
        write_bm25(root, "toy", TOY_DOCS)
        write_tfidf(root, "toy", TOY_DOCS)
        write_embeddings(root, "toy", TOY_DOCS)
        settings = RetrievalSettings(stale_check_seconds=0.0)
        store = IndexStore(cache=CompactCache(root / "cache"), stale_check_seconds=0.0)
        self.texts = FakeTexts({k: v.upper() for k, v in TOY_DOCS.items()})
        service = RetrievalService(settings, store, FakeEncoder(), self.texts, preprocess=tokens_of)
        self.addCleanup(service.close)
        main.app.state.service = service
        self.client = TestClient(main.app, raise_server_exceptions=False)

    def post(self, path, **body):
        body.setdefault("dataset", "toy")
        return self.client.post(path, json=body)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "service": "retrieval"})

    def test_retrieve_response_shape(self):
        response = self.post("/retrieve", model="bm25", query_tokens=["boundary", "layer"], top_k=3, include_text=False)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["model"], body["total"], len(body["results"])), ("bm25", 3, 3))
        self.assertEqual(set(body["results"][0]), {"doc_id", "score"})  # no "text" key when it was not requested
        self.assertIn("took_ms", body)
        self.assertNotIn("warnings", body)

    def test_include_text(self):
        body = self.post("/retrieve/bm25", query_tokens=["flutter"], include_text=True).json()
        self.assertTrue(all(r["text"] == TOY_DOCS[r["doc_id"]].upper() for r in body["results"]))

    def test_model_specific_routes_ignore_the_model_field(self):
        for model in ("tfidf", "bm25", "embedding"):
            body = self.post(f"/retrieve/{model}", model="bm25", query_tokens=["wing", "flutter"], query_text="wing flutter").json()
            self.assertEqual(body["model"], model)
            self.assertTrue(body["results"])

    def test_text_only_request_like_the_rag_service_sends(self):
        response = self.post("/retrieve", model="bm25", query_tokens=[], query_text="wing flutter")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["results"])

    def test_hybrid_serial_and_parallel(self):
        serial = self.post("/retrieve/hybrid/serial", query_text="boundary layer", query_tokens=["boundary", "layer"],
                           first_model="bm25", second_model="embedding", first_stage_k=50, final_top_k=3)
        self.assertEqual(serial.status_code, 200)
        self.assertEqual(serial.json()["model"], "hybrid_serial_bm25_embedding")
        parallel = self.post("/retrieve/hybrid/parallel", query_text="boundary layer", query_tokens=["boundary", "layer"],
                             models=["bm25", "tfidf"], top_k=3)
        self.assertEqual(parallel.status_code, 200)
        self.assertEqual(parallel.json()["model"], "hybrid_parallel_bm25_tfidf_rrf")
        self.assertLessEqual(len(parallel.json()["results"]), 3)

    def test_error_status_codes(self):
        self.assertEqual(self.post("/retrieve", model="splade", query_tokens=["a"]).status_code, 422)
        self.assertEqual(self.post("/retrieve", query_tokens=["a"], dataset=None).status_code, 400)
        self.assertEqual(self.post("/retrieve", query_tokens=[], query_text="").status_code, 400)
        self.assertEqual(self.post("/retrieve", query_tokens=["a"], dataset="missing").status_code, 404)

    def test_status_endpoints(self):
        overall = self.client.get("/status").json()
        self.assertEqual(overall["service"], "retrieval")
        self.assertEqual(overall["models"], {"tfidf": "ready", "embedding": "ready", "bm25": "ready"})  # what System Monitor reads
        self.assertEqual(self.client.get("/status", params={"dataset": "toy"}).json()["models"]["tfidf"], "ready")
        self.assertEqual(self.client.get("/status", params={"dataset": "none"}).json()["models"]["tfidf"], "not_built")


if __name__ == "__main__":
    unittest.main()
