"""HTTP contract of the ranking service (needs fastapi + httpx + pydantic)."""
import unittest

try:
    from fastapi.testclient import TestClient

    from services.ranking import main
    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover
    HAVE_FASTAPI = False


@unittest.skipUnless(HAVE_FASTAPI, "fastapi / httpx / pydantic are not installed")
class HttpTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app, raise_server_exceptions=False)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "service": "ranking"})

    def test_rank_orders_and_passes_every_field_through(self):
        results = [{"doc_id": "a", "score": 0.2, "text": None, "metadata": {"k": 1}}, {"doc_id": "b", "score": 0.9, "text": "hello"}]
        response = self.client.post("/rank", json={"results": results, "top_k": 5})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([r["doc_id"] for r in body["ranked_results"]], ["b", "a"])
        self.assertIsNone(body["ranked_results"][1]["text"])  # a null field is not dropped
        self.assertEqual(body["ranked_results"][1]["metadata"], {"k": 1})
        self.assertEqual(body["total"], 2)
        self.assertIn("took_ms", body)

    def test_rank_defaults_and_top_k(self):
        results = [{"doc_id": str(i), "score": float(i)} for i in range(100)]
        self.assertEqual(self.client.post("/rank", json={"results": results}).json()["total"], 50)
        self.assertEqual(self.client.post("/rank", json={"results": results, "top_k": 3, "method": "score"}).json()["total"], 3)

    def test_rank_errors(self):
        self.assertEqual(self.client.post("/rank", json={"results": [{"score": 1}], "method": "rrf"}).status_code, 400)
        self.assertEqual(self.client.post("/rank", json={"results": [{"doc_id": "x"}]}).status_code, 400)
        self.assertEqual(self.client.post("/rank", json={"results": [], "top_k": 0}).status_code, 422)
        self.assertEqual(self.client.post("/rank", json={}).status_code, 422)

    def test_fuse_rrf(self):
        lists = [[{"doc_id": "a", "score": 3}, {"doc_id": "b", "score": 2}], [{"doc_id": "b", "score": 9}, {"doc_id": "c", "score": 1}]]
        response = self.client.post("/fuse", json={"result_lists": lists, "method": "rrf", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([r["doc_id"] for r in body["fused_results"]], ["b", "a"])
        self.assertAlmostEqual(body["fused_results"][0]["score"], 1 / 62 + 1 / 61, places=12)
        self.assertEqual((body["method"], body["total"]), ("rrf", 2))

    def test_fuse_weighted_and_errors(self):
        lists = [[{"doc_id": "a", "score": 3}, {"doc_id": "b", "score": 1}], [{"doc_id": "b", "score": 5}, {"doc_id": "a", "score": 1}]]
        ok = self.client.post("/fuse", json={"result_lists": lists, "method": "weighted", "weights": [0.9, 0.1]})
        self.assertEqual([r["doc_id"] for r in ok.json()["fused_results"]], ["a", "b"])
        self.assertEqual(self.client.post("/fuse", json={"result_lists": lists, "method": "weighted"}).status_code, 422)
        self.assertEqual(self.client.post("/fuse", json={"result_lists": lists[:1]}).status_code, 422)
        self.assertEqual(self.client.post("/fuse", json={"result_lists": lists, "method": "borda"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
