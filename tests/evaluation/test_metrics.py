"""The metrics formulas, checked against hand-computed values."""
import math
import unittest

from services.evaluation.engines.metrics_calculator import MetricsCalculator, relevant_count


class MetricsTests(unittest.TestCase):
    def test_ndcg_of_a_perfect_ranking_is_one(self):
        qrels = {"d1": 1, "d2": 1}
        self.assertAlmostEqual(
            MetricsCalculator.calculate_ndcg(["d1", "d2"], qrels, 10), 1.0
        )

    def test_ndcg_of_a_reversed_ranking_matches_the_formula(self):
        qrels = {"d1": 1, "d2": 1}
        dcg = 1 / math.log2(3) + 1 / math.log2(2)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(
            MetricsCalculator.calculate_ndcg(["d2", "d1"], qrels, 10), dcg / idcg
        )

    def test_ndcg_uses_graded_relevance(self):
        """A grade-2 document is worth three times a grade-1 one (2^rel - 1)."""
        qrels = {"d1": 2, "d2": 1}
        self.assertAlmostEqual(MetricsCalculator.calculate_ndcg(["d1", "d2"], qrels, 10), 1.0)
        graded = MetricsCalculator.calculate_ndcg(["d2", "d1"], qrels, 10)
        self.assertLess(graded, 1.0)

    def test_ndcg_respects_the_cutoff(self):
        """With k=1 only the top document counts: a relevant doc at rank 2 earns nothing."""
        qrels = {"d1": 1}
        self.assertAlmostEqual(MetricsCalculator.calculate_ndcg(["d2", "d1"], qrels, 1), 0.0)
        # at k=2 the hit sits at rank 2 while the ideal places it at rank 1
        self.assertAlmostEqual(MetricsCalculator.calculate_ndcg(["d2", "d1"], qrels, 2), 1 / math.log2(3))

    def test_average_precision_of_a_known_ranking(self):
        qrels = {"d1": 1, "d3": 1}
        # hits at rank 1 and 3: (1/1 + 2/3) / 2
        self.assertAlmostEqual(
            MetricsCalculator.calculate_average_precision(["d1", "d2", "d3"], qrels),
            (1.0 + 2 / 3) / 2,
        )

    def test_average_precision_is_zero_without_relevant_documents(self):
        self.assertEqual(MetricsCalculator.calculate_average_precision(["d1"], {"d1": 0}), 0.0)

    def test_recall_and_precision(self):
        qrels = {"d1": 1, "d2": 1, "d3": 1}
        retrieved = ["d1", "x", "d2", "y"]
        self.assertAlmostEqual(MetricsCalculator.calculate_recall(retrieved, qrels), 2 / 3)
        self.assertAlmostEqual(MetricsCalculator.calculate_precision(retrieved, qrels), 0.5)

    def test_empty_retrieval_never_divides_by_zero(self):
        qrels = {"d1": 1}
        metrics = MetricsCalculator.calculate_all_metrics([], qrels, 10)
        self.assertEqual(metrics, {"ndcg": 0.0, "ap": 0.0, "recall": 0.0, "precision": 0.0})

    def test_empty_qrels_never_divides_by_zero(self):
        metrics = MetricsCalculator.calculate_all_metrics(["d1"], {}, 10)
        self.assertEqual(metrics["ap"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)

    def test_relevant_count_ignores_zero_and_negative_grades(self):
        self.assertEqual(relevant_count({"a": 1, "b": 0, "c": -1, "d": 2}), 2)


if __name__ == "__main__":
    unittest.main()
