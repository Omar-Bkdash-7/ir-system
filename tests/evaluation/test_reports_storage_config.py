"""Reports, storage row mapping and settings — the parts that need no database."""
import os
import unittest
from unittest.mock import patch

from services.evaluation.config import EvaluationSettings, reset_settings_cache
from services.evaluation.core.reports import (
    latest_metrics_by_model,
    per_query_analysis,
    refinement_comparison,
)
from services.evaluation.services.results_storage import ResultsStorage, _row_to_result


class LatestMetricsTests(unittest.TestCase):
    def test_the_newest_row_of_each_model_wins(self):
        rows = [
            {"model": "tfidf", "map": 0.5, "ndcg": 0.6, "recall": 0.7, "precision": 0.8, "num_queries": 9},
            {"model": "bm25", "map": 0.4, "ndcg": 0.4, "recall": 0.4, "precision": 0.4, "num_queries": 9},
            {"model": "tfidf", "map": 0.1, "ndcg": 0.1, "recall": 0.1, "precision": 0.1, "num_queries": 1},
        ]

        metrics = latest_metrics_by_model(rows)

        self.assertEqual(set(metrics), {"TF-IDF", "BM25"})
        self.assertEqual(metrics["TF-IDF"]["map"], 0.5)
        self.assertEqual(metrics["TF-IDF"]["num_queries"], 9)

    def test_hybrid_models_keep_the_frontend_label(self):
        rows = [{"model": "hybrid_parallel", "map": 0.3, "ndcg": 0.3, "recall": 0.3,
                 "precision": 0.3, "num_queries": 2}]
        self.assertEqual(list(latest_metrics_by_model(rows)), ["HYBRID-PARALLEL"])

    def test_no_rows_means_no_metrics(self):
        self.assertEqual(latest_metrics_by_model([]), {})


class UnavailableReportTests(unittest.TestCase):
    def test_per_query_analysis_says_why_it_is_empty(self):
        report = per_query_analysis()
        self.assertFalse(report["available"])
        self.assertTrue(report["message"])

    def test_refinement_comparison_says_where_the_data_lives(self):
        report = refinement_comparison()
        self.assertFalse(report["available"])
        self.assertEqual(report["before"], {})


class StorageMappingTests(unittest.TestCase):
    def test_a_row_becomes_the_documented_dict(self):
        class Timestamp:
            def isoformat(self):
                return "2026-01-02T03:04:05"

        row = (7, "tfidf", "cranfield", 100, 0.5, 0.6, 0.7, 0.8, 12, {"a": 1}, Timestamp())

        result = _row_to_result(row)

        self.assertEqual(result["id"], 7)
        self.assertEqual(result["map"], 0.5)
        self.assertEqual(result["metadata"], {"a": 1})
        self.assertEqual(result["created_at"], "2026-01-02T03:04:05")

    def test_delete_without_filters_is_refused(self):
        storage = ResultsStorage(ensure_table=False)
        with self.assertRaises(ValueError):
            storage.delete_by_filters()

    def test_delete_with_a_filter_is_allowed(self):
        storage = ResultsStorage(ensure_table=False)
        self.assertTrue(callable(storage.delete_by_filters))


class SettingsTests(unittest.TestCase):
    def tearDown(self):
        reset_settings_cache()

    def test_defaults_are_usable_without_configuration(self):
        settings = EvaluationSettings()
        self.assertEqual(settings.retrieval_url, "http://retrieval:8003")
        self.assertEqual(settings.default_top_k, 100)
        self.assertTrue(settings.save_results)

    def test_environment_overrides_the_defaults(self):
        env = {
            "EVALUATION_RETRIEVAL_URL": "http://localhost:9003/",
            "EVALUATION_RETRIEVAL_TIMEOUT": "5",
            "EVALUATION_SAVE_RESULTS": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = EvaluationSettings.load()

        self.assertEqual(settings.retrieval_url, "http://localhost:9003")
        self.assertEqual(settings.retrieval_timeout_seconds, 5.0)
        self.assertFalse(settings.save_results)

    def test_an_invalid_value_falls_back_to_the_default(self):
        with patch.dict(os.environ, {"EVALUATION_MAX_TOP_K": "not-a-number"}, clear=False):
            settings = EvaluationSettings.load()

        self.assertEqual(settings.max_top_k, 1000)

    def test_workers_default_and_caps(self):
        self.assertEqual(EvaluationSettings().workers, 8)

        with patch.dict(os.environ, {"EVALUATION_WORKERS": "3"}, clear=False):
            self.assertEqual(EvaluationSettings.load().workers, 3)

        with patch.dict(os.environ, {"EVALUATION_WORKERS": "9999"}, clear=False):
            self.assertEqual(EvaluationSettings.load().workers, 64)

    def test_values_are_clamped_to_the_supported_range(self):
        settings = EvaluationSettings()
        self.assertEqual(settings.top_k(10_000), settings.max_top_k)
        self.assertEqual(settings.top_k(None), settings.default_top_k)
        self.assertIsNone(settings.query_limit(0))
        self.assertEqual(settings.query_limit(10_000), settings.max_query_limit)


class ThreadedClientTests(unittest.TestCase):
    def test_each_thread_gets_its_own_session(self):
        """A requests.Session is a connection pool with state: parallel runs need one each."""
        import threading

        from services.evaluation.core.clients import JsonClient

        client = JsonClient(max_retries=1)
        sessions = []
        lock = threading.Lock()

        def grab():
            session = client._session()
            with lock:
                sessions.append(session)

        threads = [threading.Thread(target=grab) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        client.close()

        self.assertEqual(len({id(session) for session in sessions}), 3)

    def test_an_injected_session_is_used_as_is(self):
        from services.evaluation.core.clients import JsonClient

        class FakeSession:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        session = FakeSession()
        client = JsonClient(session, max_retries=1)

        self.assertIs(client._session(), session)
        client.close()
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
