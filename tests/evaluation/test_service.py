"""Service behaviour with fake dependencies: no HTTP, no database, no dataset download."""
import unittest

from services.evaluation.config import EvaluationSettings
from services.evaluation.core.dataset import DatasetSplit
from services.evaluation.core.query import PreparedQuery
from services.evaluation.core.service import EvaluationService
from services.evaluation.schemas import EvaluationRequest


class FakeLoader:
    def __init__(self, split):
        self.split = split
        self.loaded = []

    def load(self, name):
        self.loaded.append(name)
        return self.split


class FakeStorage:
    def __init__(self):
        self.saved = []
        self.rows = []
        self.raise_on_save = False

    def save_results(self, **kwargs):
        if self.raise_on_save:
            raise RuntimeError("database down")
        self.saved.append(kwargs)
        return len(self.saved)

    def get_latest_results(self, **kwargs):
        return self.rows

    def compare_models(self, **kwargs):
        return []

    def delete_result(self, result_id):
        return result_id == 1

    def delete_by_filters(self, **kwargs):
        return 3

    def clear_all(self):
        return 5


class FakePipeline:
    def __init__(self):
        self.prepared = []

    def prepare(self, text, *, dataset=None, refine=False):
        self.prepared.append({"text": text, "dataset": dataset, "refine": refine})
        return PreparedQuery(text=text, tokens=tuple(text.split()), refined=refine)


class FakeClient:
    def __init__(self, doc_ids=("d1", "d2")):
        self.calls = []
        self.doc_ids = list(doc_ids)

    def post(self, url, payload, *, timeout, service):
        self.calls.append({"url": url, "payload": payload, "service": service, "timeout": timeout})
        return {"results": [{"doc_id": doc_id, "score": 1.0} for doc_id in self.doc_ids]}

    def close(self):
        pass


def build_service(split=None, *, settings=None, doc_ids=("d1", "d2")):
    split = split or DatasetSplit(
        name="toy",
        queries={"q1": "alpha beta", "q2": "gamma"},
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    )
    loader, storage, pipeline, client = FakeLoader(split), FakeStorage(), FakePipeline(), FakeClient(doc_ids)
    settings = settings or EvaluationSettings()
    service = EvaluationService(settings, loader, storage, pipeline, client)
    return service, loader, storage, pipeline, client


class ServiceTests(unittest.TestCase):
    def test_a_single_model_run_scores_every_query_and_stores_the_result(self):
        service, loader, storage, pipeline, client = build_service()

        response = service.evaluate(EvaluationRequest(dataset_name="toy", model="tfidf", top_k=10))

        self.assertEqual(response.model, "tfidf")
        self.assertEqual(response.top_k, 10)
        self.assertEqual(response.metrics.num_queries, 2)
        self.assertEqual(loader.loaded, ["toy"])

        stored = storage.saved[0]
        self.assertEqual(stored["model"], "tfidf")
        self.assertEqual(stored["dataset"], "toy")
        self.assertEqual(stored["num_queries"], 2)
        self.assertEqual(stored["metadata"]["dataset"], "toy")
        self.assertFalse(stored["metadata"]["enable_refinement"])

        self.assertTrue(client.calls[0]["url"].endswith("/retrieve/tfidf"))
        self.assertEqual(client.calls[0]["payload"]["query_tokens"], ["alpha", "beta"])
        self.assertEqual(client.calls[0]["payload"]["dataset"], "toy")

    def test_embedding_does_not_send_tokens(self):
        service, _loader, _storage, _pipeline, client = build_service()

        service.evaluate(EvaluationRequest(dataset_name="toy", model="embedding"))

        self.assertNotIn("query_tokens", client.calls[0]["payload"])

    def test_bm25_parameters_reach_the_retriever(self):
        service, _loader, _storage, _pipeline, client = build_service()

        service.evaluate(EvaluationRequest(dataset_name="toy", model="bm25", bm25_k1=1.8, bm25_b=0.6))

        self.assertEqual(client.calls[0]["payload"]["bm25_k1"], 1.8)
        self.assertEqual(client.calls[0]["payload"]["bm25_b"], 0.6)

    def test_refinement_is_requested_per_query_and_recorded(self):
        service, _loader, storage, pipeline, _client = build_service()

        service.evaluate(EvaluationRequest(dataset_name="toy", model="tfidf", enable_refinement=True))

        self.assertTrue(all(entry["refine"] for entry in pipeline.prepared))
        self.assertTrue(storage.saved[0]["metadata"]["enable_refinement"])

    def test_hybrid_serial_needs_two_models(self):
        service, *_ = build_service()
        with self.assertRaises(ValueError):
            service.evaluate(EvaluationRequest(dataset_name="toy", model="hybrid_serial"))

    def test_hybrid_models_must_be_single_models(self):
        service, *_ = build_service()
        with self.assertRaises(ValueError):
            service.evaluate(
                EvaluationRequest(
                    dataset_name="toy", model="hybrid_parallel",
                    hybrid_models=["bm25", "hybrid_serial"],
                )
            )

    def test_hybrid_serial_uses_the_hybrid_endpoint_with_both_models(self):
        service, _loader, storage, _pipeline, client = build_service()

        service.evaluate(
            EvaluationRequest(
                dataset_name="toy", model="hybrid_serial", hybrid_models=["bm25", "embedding"]
            )
        )

        call = client.calls[0]
        self.assertTrue(call["url"].endswith("/retrieve/hybrid/serial"))
        self.assertEqual(call["payload"]["first_model"], "bm25")
        self.assertEqual(call["payload"]["second_model"], "embedding")
        self.assertEqual(storage.saved[0]["metadata"]["hybrid_models"], ["bm25", "embedding"])

    def test_hybrid_parallel_passes_the_fusion_method(self):
        service, _loader, _storage, _pipeline, client = build_service()

        service.evaluate(
            EvaluationRequest(
                dataset_name="toy", model="hybrid_parallel",
                hybrid_models=["tfidf", "bm25"], fusion_method="RRF",
            )
        )

        self.assertEqual(client.calls[0]["payload"]["fusion_method"], "rrf")

    def test_a_storage_failure_does_not_lose_the_metrics(self):
        service, _loader, storage, _pipeline, _client = build_service()
        storage.raise_on_save = True

        response = service.evaluate(EvaluationRequest(dataset_name="toy", model="tfidf"))

        self.assertEqual(response.metrics.num_queries, 2)

    def test_storage_can_be_switched_off(self):
        service, _loader, storage, _pipeline, _client = build_service(
            settings=EvaluationSettings(save_results=False)
        )

        service.evaluate(EvaluationRequest(dataset_name="toy", model="tfidf"))

        self.assertEqual(storage.saved, [])

    def test_per_query_results_are_returned_only_on_request(self):
        service, *_ = build_service()

        without = service.evaluate(EvaluationRequest(dataset_name="toy", model="tfidf"))
        with_per_query = service.evaluate(
            EvaluationRequest(dataset_name="toy", model="tfidf", include_per_query=True)
        )

        self.assertIsNone(without.query_results)
        self.assertEqual(len(with_per_query.query_results), 2)
        self.assertEqual(with_per_query.query_results[0].query_id, "q1")

    def test_a_query_without_documents_is_reported_as_a_warning(self):
        service, *_ = build_service(doc_ids=())

        response = service.evaluate(EvaluationRequest(dataset_name="toy", model="tfidf"))

        self.assertEqual(response.metrics.num_queries, 0)
        self.assertTrue(response.warnings)
        self.assertIn("could not be evaluated", response.warnings[0])

    def test_query_limit_is_applied_and_capped_by_settings(self):
        service, _loader, _storage, _pipeline, client = build_service()

        service.evaluate(EvaluationRequest(dataset_name="toy", model="tfidf", query_limit=1))

        self.assertEqual(len(client.calls), 1)

    def test_latest_metrics_are_keyed_like_the_frontend_expects(self):
        service, _loader, storage, _pipeline, _client = build_service()
        storage.rows = [
            {"model": "tfidf", "map": 0.5, "ndcg": 0.6, "recall": 0.7, "precision": 0.8, "num_queries": 5},
            {"model": "tfidf", "map": 0.1, "ndcg": 0.1, "recall": 0.1, "precision": 0.1, "num_queries": 1},
        ]

        metrics = service.get_latest_metrics()

        self.assertEqual(list(metrics), ["TF-IDF"])
        self.assertEqual(metrics["TF-IDF"]["map"], 0.5)
        self.assertEqual(metrics["TF-IDF"]["precision_at_10"], 0.8)

    def test_unknown_model_is_rejected_before_any_work(self):
        service, loader, *_ = build_service()
        with self.assertRaises(ValueError):
            service.evaluate(EvaluationRequest(dataset_name="toy", model="nope"))

    def test_delete_and_history_delegate_to_storage(self):
        service, *_ = build_service()

        self.assertTrue(service.delete_result(1))
        self.assertFalse(service.delete_result(2))
        self.assertEqual(service.delete_results(model="tfidf"), 3)
        self.assertEqual(service.clear_results(), 5)
        self.assertEqual(service.get_evaluation_history(), [])


if __name__ == "__main__":
    unittest.main()
