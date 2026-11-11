"""HTTP contract of the evaluation service: routes, status codes and JSON shapes."""
import unittest

try:
    from fastapi.testclient import TestClient

    from services.evaluation import main

    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover - the API tests simply do not run
    HAVE_FASTAPI = False


class FakeService:
    """Records what the endpoints pass through, so the HTTP layer can be checked alone."""

    def __init__(self):
        self.evaluated = []

    def evaluate(self, request):
        self.evaluated.append(request)
        return {
            "model": request.model,
            "top_k": request.top_k,
            "metrics": {"map": 0.5, "ndcg": 0.6, "recall": 0.7, "precision": 0.8, "num_queries": 3},
        }

    def get_available_datasets(self):
        return ["cranfield"]

    def get_evaluation_history(self, model=None, dataset=None, limit=10):
        return []

    def compare_models(self, dataset, top_k=100):
        return []

    def get_latest_metrics(self):
        return {"TF-IDF": {"map": 0.5, "ndcg_at_10": 0.6, "recall": 0.7, "precision_at_10": 0.8, "num_queries": 3}}

    def get_per_query_analysis(self):
        return {"message": "not implemented", "available": False}

    def get_refinement_comparison(self):
        return {"message": "not implemented", "available": False, "before": {}, "after": {}}

    def delete_result(self, result_id):
        return result_id == 5

    def delete_results(self, model=None, dataset=None, before_date=None):
        return 4

    def clear_results(self):
        return 9

    def close(self):  # pragma: no cover - the lifespan is not run by TestClient here
        pass


@unittest.skipUnless(HAVE_FASTAPI, "fastapi / httpx / pydantic are not installed")
class HttpTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        main.app.state.service = self.service
        self.client = TestClient(main.app, raise_server_exceptions=False)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "service": "evaluation"})

    def test_status_lists_models_and_metrics(self):
        body = self.client.get("/status").json()
        self.assertEqual(body["service"], "evaluation")
        self.assertIn("hybrid_parallel", body["supported_models"])
        self.assertIn("MAP", body["supported_metrics"])

    def test_datasets_are_served_from_the_service(self):
        self.assertEqual(self.client.get("/datasets").json(), {"datasets": ["cranfield"]})

    def test_evaluate_returns_metrics(self):
        response = self.client.post("/evaluate", json={"dataset_name": "toy", "model": "bm25", "top_k": 20})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model"], "bm25")
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["metrics"]["num_queries"], 3)
        self.assertEqual(self.service.evaluated[0].dataset_name, "toy")

    def test_an_unknown_model_is_a_validation_error(self):
        response = self.client.post("/evaluate", json={"dataset_name": "toy", "model": "nope"})
        self.assertEqual(response.status_code, 422)

    def test_a_rejected_run_is_a_client_error(self):
        self.service.evaluate = lambda request: (_ for _ in ()).throw(ValueError("dataset has no qrels"))
        response = self.client.post("/evaluate", json={"dataset_name": "toy", "model": "tfidf"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("dataset has no qrels", response.text)

    def test_run_accepts_query_parameters(self):
        response = self.client.post("/run", params={"model": "tfidf", "dataset": "toy", "top_k": 5})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.evaluated[0].top_k, 5)

    def test_history_and_compare_shapes(self):
        self.assertEqual(self.client.get("/history").json(), {"history": []})

        compare = self.client.get("/compare", params={"dataset": "toy"}).json()
        self.assertEqual(compare["dataset"], "toy")
        self.assertEqual(compare["models"], [])

    def test_metrics_are_keyed_by_model_label(self):
        body = self.client.get("/metrics").json()
        self.assertEqual(list(body), ["TF-IDF"])
        self.assertEqual(body["TF-IDF"]["precision_at_10"], 0.8)

    def test_deleting_a_missing_result_is_a_404(self):
        self.assertEqual(self.client.delete("/result/5").status_code, 200)
        response = self.client.delete("/result/6")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])

    def test_bulk_delete_requires_a_filter(self):
        self.assertEqual(self.client.delete("/results").status_code, 400)

        response = self.client.delete("/results", params={"model": "tfidf"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 4)

    def test_clear_all_reports_the_count(self):
        body = self.client.delete("/results/clear-all").json()
        self.assertEqual(body["count"], 9)
        self.assertTrue(body["success"])


if __name__ == "__main__":
    unittest.main()
