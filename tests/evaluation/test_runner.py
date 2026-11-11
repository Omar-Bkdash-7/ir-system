"""The shared run loop: scoring, skipping and surviving broken queries."""
import unittest

from services.evaluation.core.dataset import DatasetSplit
from services.evaluation.core.query import PreparedQuery
from services.evaluation.core.runner import aggregate_metrics, run_over_queries


def split(**overrides):
    defaults = {
        "name": "toy",
        "queries": {"q1": "alpha beta", "q2": "gamma", "q3": "ungraded"},
        "qrels": {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}},
    }
    defaults.update(overrides)
    return DatasetSplit(**defaults)


def prepare(text):
    return PreparedQuery(text=text, tokens=tuple(text.split()))


def retrieved(pairs):
    return [{"doc_id": doc_id, "score": 1.0} for doc_id in pairs]


class RunTests(unittest.TestCase):
    def test_every_evaluable_query_is_scored_and_averaged(self):
        run = run_over_queries(
            split(),
            prepare=prepare,
            retrieve=lambda query: retrieved(["d1", "d3"]),
            top_k=10,
        )

        self.assertEqual(set(run.results), {"q1", "q2"})
        self.assertEqual(run.metrics["num_queries"], 2)
        self.assertEqual(run.failures, [])
        # q1 retrieves one of two relevant docs (recall .5), q2 both of one (1.0)
        self.assertAlmostEqual(run.metrics["recall"], 0.75)

    def test_a_query_without_qrels_is_skipped_not_failed(self):
        run = run_over_queries(split(), prepare=prepare, retrieve=lambda q: retrieved(["d1"]), top_k=10)
        self.assertNotIn("q3", run.results)
        self.assertEqual(run.failures, [])

    def test_one_broken_query_does_not_stop_the_run(self):
        def retrieve(query):
            if query.text == "gamma":
                raise RuntimeError("retrieval exploded")
            return retrieved(["d1"])

        run = run_over_queries(split(), prepare=prepare, retrieve=retrieve, top_k=10)

        self.assertEqual(set(run.results), {"q1"})
        self.assertEqual(len(run.failures), 1)
        self.assertEqual(run.failures[0]["query_id"], "q2")
        self.assertIn("retrieval exploded", run.failures[0]["error"])
        self.assertEqual(run.metrics["num_queries"], 1)

    def test_an_empty_result_is_reported_as_a_failure(self):
        run = run_over_queries(split(), prepare=prepare, retrieve=lambda q: [], top_k=10)
        self.assertEqual(len(run.failures), 2)
        self.assertEqual(run.failures[0]["error"], "no documents retrieved")
        self.assertEqual(run.metrics["num_queries"], 0)

    def test_warnings_summarise_the_failures(self):
        run = run_over_queries(split(), prepare=prepare, retrieve=lambda q: [], top_k=10)
        warnings = run.warnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn("2 query", warnings[0])

    def test_limit_keeps_only_evaluable_queries(self):
        limited = split().limited(1)
        self.assertEqual(limited.evaluable_queries, ["q1"])
        self.assertEqual(len(limited.queries), 1)

    def test_aggregate_of_nothing_is_all_zero(self):
        metrics = aggregate_metrics([])
        self.assertEqual(metrics, {"map": 0.0, "ndcg": 0.0, "recall": 0.0, "precision": 0.0, "num_queries": 0})


class ParallelRunTests(unittest.TestCase):
    """Workers change the speed, never the numbers."""

    def test_parallel_and_sequential_runs_agree(self):
        retrieve = lambda query: retrieved(["d1", "d3"])

        sequential = run_over_queries(split(), prepare=prepare, retrieve=retrieve, top_k=10, workers=1)
        parallel = run_over_queries(split(), prepare=prepare, retrieve=retrieve, top_k=10, workers=4)

        self.assertEqual(sequential.metrics, parallel.metrics)
        self.assertEqual(sequential.results, parallel.results)

    def test_results_keep_dataset_order_whatever_the_completion_order(self):
        import time as _time

        def retrieve(query):
            # q1 finishes last, so completion order would be q2, q1
            _time.sleep(0.02 if query.text == "alpha beta" else 0.0)
            return retrieved(["d1"])

        run = run_over_queries(split(), prepare=prepare, retrieve=retrieve, top_k=10, workers=2)

        self.assertEqual(list(run.results), ["q1", "q2"])

    def test_a_broken_query_is_isolated_from_the_others(self):
        def retrieve(query):
            if query.text == "gamma":
                raise RuntimeError("retrieval exploded")
            return retrieved(["d1"])

        run = run_over_queries(split(), prepare=prepare, retrieve=retrieve, top_k=10, workers=4)

        self.assertEqual(set(run.results), {"q1"})
        self.assertEqual([failure["query_id"] for failure in run.failures], ["q2"])

    def test_parallel_run_works_with_a_single_query(self):
        run = run_over_queries(
            split().limited(1), prepare=prepare, retrieve=lambda q: retrieved(["d1"]), top_k=10, workers=8
        )

        self.assertEqual(run.metrics["num_queries"], 1)
        self.assertEqual(run.failures, [])

    def test_workers_speed_up_a_run_of_slow_queries(self):
        """The point of the feature: N slow queries finish in about one query's time."""
        import time as _time

        def retrieve(query):
            _time.sleep(0.05)
            return retrieved(["d1"])

        big = DatasetSplit(
            name="toy",
            queries={f"q{i}": "alpha" for i in range(8)},
            qrels={f"q{i}": {"d1": 1} for i in range(8)},
        )

        started = _time.perf_counter()
        run_over_queries(big, prepare=prepare, retrieve=retrieve, top_k=10, workers=8)
        elapsed = _time.perf_counter() - started

        self.assertLess(elapsed, 0.2, "8 queries of 50ms should finish in about 50ms, not 400ms")


if __name__ == "__main__":
    unittest.main()
