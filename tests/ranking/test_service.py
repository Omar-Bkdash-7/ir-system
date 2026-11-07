import logging
import unittest

from common.exceptions import InvalidQueryError
from services.ranking.ranking_service import fuse_result_lists, rank_results
from services.ranking.rankers import fuse_results


def hits(*pairs):
    return [{"doc_id": d, "score": s} for d, s in pairs]


class RankResultsTests(unittest.TestCase):
    def test_orders_and_keeps_every_field(self):
        data = [{"doc_id": "a", "score": 0.2, "text": "ta", "metadata": {"k": 1}, "extra": None}, {"doc_id": "b", "score": 0.9}]
        out = rank_results(data, "cosine", 10)
        self.assertEqual([r["doc_id"] for r in out["ranked_results"]], ["b", "a"])
        self.assertIs(out["ranked_results"][1], data[0])  # same object, nothing dropped (not even the null field)
        self.assertEqual((out["total"], out["took_ms"] >= 0), (2, True))

    def test_top_k_and_ties(self):
        data = [{"doc_id": str(i), "score": 1.0} for i in range(10)]
        out = rank_results(data, "score", 3)
        self.assertEqual([r["doc_id"] for r in out["ranked_results"]], ["0", "1", "2"])

    def test_method_names(self):
        data = [{"score": 1}, {"score": 2}]
        for name in ("cosine", "score", "COSINE", " Score "):
            self.assertEqual(rank_results(data, name, 5)["ranked_results"][0]["score"], 2)
        with self.assertRaisesRegex(InvalidQueryError, "Unknown ranking method: rrf"):
            rank_results(data, "rrf", 5)

    def test_a_missing_or_invalid_score_is_a_bad_request(self):
        for bad in ({"doc_id": "x"}, {"score": None}, {"score": "0.5"}, {"score": True}, {"score": [1]}):
            with self.assertRaisesRegex(InvalidQueryError, r"results\[1\] needs a numeric 'score'"):
                rank_results([{"score": 1}, bad], "cosine", 5)
        with self.assertRaisesRegex(InvalidQueryError, r"results\[0\]"):
            rank_results(["not a dict"], "cosine", 5)

    def test_non_finite_scores_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(InvalidQueryError, r"results\[1\]: 'score' must be a finite number"):
                rank_results([{"score": 1}, {"score": value}], "cosine", 5)

    def test_empty_results(self):
        out = rank_results([], "cosine", 5)
        self.assertEqual((out["ranked_results"], out["total"]), ([], 0))


class FuseResultListsTests(unittest.TestCase):
    A = hits(("a", 3.0), ("b", 2.0), ("c", 1.0))
    B = hits(("c", 0.9), ("a", 0.5), ("d", 0.1))

    def setUp(self):
        logging.getLogger("services.ranking.rankers.score_fusion").setLevel(logging.ERROR)

    def test_response_and_method_echo(self):
        out = fuse_result_lists([self.A, self.B], "RRF", None, 60, 3)
        self.assertEqual((out["method"], out["total"], len(out["fused_results"])), ("rrf", 3, 3))
        self.assertEqual(out["fused_results"], fuse_results([self.A, self.B], "rrf", None, 60)[:3])

    def test_lists_are_ordered_by_score_before_they_are_fused(self):
        shuffled = [self.A[2], self.A[0], self.A[1]]
        self.assertEqual(fuse_result_lists([shuffled, self.B], "rrf")["fused_results"],
                         fuse_result_lists([self.A, self.B], "rrf")["fused_results"])

    def test_equal_scores_in_a_list_keep_the_order_sent(self):
        flat = hits(("x", 1.0), ("y", 1.0), ("z", 1.0))
        out = fuse_result_lists([flat, hits(("q", 2.0))], "rrf")
        self.assertEqual([r["doc_id"] for r in out["fused_results"]], ["x", "q", "y", "z"])

    def test_every_method(self):
        for method, weights in (("rrf", None), ("combsum", None), ("combmnz", None), ("weighted", [0.7, 0.3])):
            out = fuse_result_lists([self.A, self.B], method, weights, 60, 10)
            self.assertEqual(out["fused_results"], fuse_results([self.A, self.B], method, weights, 60)[:10])

    def test_errors_are_bad_requests(self):
        with self.assertRaisesRegex(InvalidQueryError, "Unknown fusion method: borda"):
            fuse_result_lists([self.A, self.B], "borda")
        with self.assertRaisesRegex(InvalidQueryError, "at least 2"):
            fuse_result_lists([self.A], "rrf")
        with self.assertRaises(ValueError):  # weights are checked by the fusion itself
            fuse_result_lists([self.A, self.B], "weighted", [1.0])


if __name__ == "__main__":
    unittest.main()
