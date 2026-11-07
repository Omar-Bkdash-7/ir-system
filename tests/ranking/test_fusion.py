import copy
import logging
import threading
import unittest

from services.ranking.rankers import (DEFAULT_RRF_K, METHODS, combmnz_fusion, combsum_fusion, fuse_results, normalize_scores,
                                      reciprocal_rank_fusion, weighted_fusion)


def hits(*pairs):
    return [{"doc_id": doc_id, "score": score} for doc_id, score in pairs]


A = hits(("a", 3.0), ("b", 2.0), ("c", 1.0))
B = hits(("c", 0.9), ("a", 0.5), ("d", 0.1))


def ids(results):
    return [item["doc_id"] for item in results]


class QuietLogs(unittest.TestCase):
    def setUp(self):
        logging.getLogger("services.ranking.rankers.score_fusion").setLevel(logging.ERROR)  # the one-list warning is expected here


class ReciprocalRankFusionTests(QuietLogs):
    def test_scores_are_the_sum_of_one_over_k_plus_rank(self):
        fused = reciprocal_rank_fusion([A, B], k=60)
        expected = {"a": 1 / 61 + 1 / 62, "b": 1 / 62, "c": 1 / 63 + 1 / 61, "d": 1 / 63}
        self.assertEqual(ids(fused), ["a", "c", "b", "d"])
        for item in fused:
            self.assertAlmostEqual(item["score"], expected[item["doc_id"]], places=15)

    def test_the_scores_themselves_are_ignored(self):
        scaled = [[{"doc_id": h["doc_id"], "score": h["score"] * 1000 + 7} for h in lst] for lst in (A, B)]
        self.assertEqual(reciprocal_rank_fusion(scaled), reciprocal_rank_fusion([A, B]))

    def test_k_changes_the_scores_not_the_documents(self):
        for k in (0, 1, 60, 100):
            fused = reciprocal_rank_fusion([A, B], k=k)
            self.assertEqual(set(ids(fused)), {"a", "b", "c", "d"})
            self.assertAlmostEqual(fused[-1]["score"], min(fused, key=lambda i: i["score"])["score"])

    def test_zero_k_is_allowed_and_negative_k_is_not(self):
        self.assertAlmostEqual(reciprocal_rank_fusion([A, B], k=0)[0]["score"], 1.0 + 1 / 2)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([A, B], k=-1)
        with self.assertRaises(ValueError):
            fuse_results([A, B], "rrf", rrf_k=-5)

    def test_a_document_listed_twice_in_one_list_counts_twice(self):
        fused = reciprocal_rank_fusion([hits(("x", 2), ("y", 1), ("x", 0.5)), hits(("y", 1))], k=60)
        self.assertAlmostEqual({i["doc_id"]: i["score"] for i in fused}["x"], 1 / 61 + 1 / 63, places=15)


class ScoreBasedFusionTests(QuietLogs):
    def test_combsum_adds_min_max_normalised_scores(self):
        fused = {i["doc_id"]: i["score"] for i in combsum_fusion([A, B])}
        # A normalised: a=1, b=.5, c=0   B normalised: c=1, a=.5, d=0
        self.assertEqual(fused, {"a": 1.5, "b": 0.5, "c": 1.0, "d": 0.0})
        self.assertEqual(ids(combsum_fusion([A, B])), ["a", "c", "b", "d"])

    def test_combmnz_multiplies_by_the_number_of_lists_containing_the_document(self):
        fused = {i["doc_id"]: i["score"] for i in combmnz_fusion([A, B])}
        self.assertEqual(fused, {"a": 3.0, "b": 0.5, "c": 2.0, "d": 0.0})

    def test_weighted_scales_each_list(self):
        fused = {i["doc_id"]: i["score"] for i in weighted_fusion([A, B], [0.8, 0.2])}
        self.assertAlmostEqual(fused["a"], 0.8 * 1 + 0.2 * 0.5, places=15)
        self.assertAlmostEqual(fused["c"], 0.8 * 0 + 0.2 * 1, places=15)
        self.assertAlmostEqual(fused["b"], 0.4, places=15)

    def test_weights_must_match_the_lists(self):
        with self.assertRaisesRegex(ValueError, r"Number of result lists \(2\) must match number of weights \(3\)"):
            weighted_fusion([A, B], [0.2, 0.3, 0.5])

    def test_equal_scores_and_single_item_lists_normalise_to_one(self):
        flat = hits(("p", 5.0), ("q", 5.0))
        self.assertEqual([i["score"] for i in normalize_scores(flat)], [1.0, 1.0])
        self.assertEqual([i["score"] for i in normalize_scores(hits(("p", -3.0)))], [1.0])
        self.assertEqual({i["doc_id"]: i["score"] for i in combsum_fusion([flat, hits(("p", 1), ("z", 0))])}, {"p": 2.0, "q": 1.0, "z": 0.0})

    def test_normalize_scores(self):
        self.assertEqual([i["score"] for i in normalize_scores(A)], [1.0, 0.5, 0.0])
        empty = []
        self.assertIs(normalize_scores(empty), empty)

    def test_negative_scores_are_normalised_too(self):
        self.assertEqual([i["score"] for i in normalize_scores(hits(("a", -1.0), ("b", -3.0)))], [1.0, 0.0])


class FuseResultsTests(QuietLogs):
    def test_dispatch_is_case_insensitive(self):
        self.assertEqual(fuse_results([A, B], "RRF"), reciprocal_rank_fusion([A, B], k=DEFAULT_RRF_K))
        self.assertEqual(fuse_results([A, B], "CombSUM"), combsum_fusion([A, B]))
        self.assertEqual(fuse_results([A, B], "combmnz"), combmnz_fusion([A, B]))
        self.assertEqual(fuse_results([A, B], "Weighted", [0.5, 0.5]), weighted_fusion([A, B], [0.5, 0.5]))
        self.assertEqual(set(METHODS), {"rrf", "combsum", "combmnz", "weighted"})

    def test_defaults(self):
        self.assertEqual(fuse_results([A, B]), reciprocal_rank_fusion([A, B], k=60))

    def test_no_lists_gives_nothing(self):
        self.assertEqual(fuse_results([]), [])

    def test_a_single_list_comes_back_unchanged(self):
        self.assertIs(fuse_results([A]), A)
        self.assertIs(fuse_results([A], "no-such-method"), A)  # nothing to fuse, so the method is not even looked at
        self.assertEqual(fuse_results([A], top_k=2), A[:2])

    def test_empty_lists_are_fine(self):
        self.assertEqual(fuse_results([[], []]), [])
        self.assertEqual(ids(fuse_results([[], A], "combsum")), ["a", "b", "c"])
        self.assertEqual(ids(fuse_results([A, []], "rrf")), ["a", "b", "c"])
        self.assertEqual(fuse_results([[], []], "combmnz"), [])
        self.assertEqual(fuse_results([[], []], "weighted", [0.5, 0.5]), [])

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "Unknown fusion method: bogus"):
            fuse_results([A, B], "bogus")
        for weights in (None, []):
            with self.assertRaisesRegex(ValueError, "Weights are required for weighted fusion"):
                fuse_results([A, B], "weighted", weights)
        with self.assertRaises(ValueError):
            fuse_results([A, B], "weighted", [1.0])

    def test_weights_are_ignored_by_the_other_methods(self):
        self.assertEqual(fuse_results([A, B], "rrf", [9, 9, 9]), fuse_results([A, B], "rrf"))

    def test_ties_keep_the_order_of_first_appearance(self):
        first, second = hits(("x", 1.0), ("y", 0.5)), hits(("z", 1.0), ("w", 0.5))
        self.assertEqual(ids(fuse_results([first, second], "rrf")), ["x", "z", "y", "w"])
        self.assertEqual(ids(fuse_results([second, first], "rrf")), ["z", "x", "w", "y"])

    def test_top_k_keeps_the_best_fused_documents(self):
        full = fuse_results([A, B], "rrf")
        self.assertEqual(fuse_results([A, B], "rrf", top_k=2), full[:2])
        self.assertEqual(fuse_results([A, B], "rrf", top_k=100), full)

    def test_inputs_are_not_modified_and_extra_fields_are_ignored(self):
        rich = [dict(h, text="t", metadata={"k": 1}) for h in A]
        before = copy.deepcopy(rich)
        fused = fuse_results([rich, B], "combsum")
        self.assertEqual(rich, before)
        self.assertTrue(all(set(item) == {"doc_id", "score"} for item in fused))

    def test_output_is_plain_python(self):
        for item in fuse_results([A, B], "combmnz"):
            self.assertIs(type(item["score"]), float)
            self.assertIs(type(item["doc_id"]), str)

    def test_non_string_ids_are_returned_as_they_came(self):
        fused = fuse_results([hits((1, 2.0), (2, 1.0)), hits((2, 3.0), (3, 1.0))], "rrf")
        self.assertEqual(sorted(ids(fused)), [1, 2, 3])
        self.assertTrue(all(type(i) is int for i in ids(fused)))

    def test_integer_scores(self):
        self.assertEqual(ids(fuse_results([hits(("a", 3), ("b", 1)), hits(("b", 5), ("a", 2))], "combsum")), ["a", "b"])

    def test_large_lists(self):
        big = [[{"doc_id": f"d{(i * 7 + j) % 4000}", "score": float(5000 - i)} for i in range(5000)] for j in range(3)]
        for method, weights in (("rrf", None), ("combsum", None), ("combmnz", None), ("weighted", [0.5, 0.3, 0.2])):
            fused = fuse_results(big, method, weights)
            self.assertLessEqual(len(fused), 4000)
            scores = [item["score"] for item in fused]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_concurrent_calls_do_not_interfere(self):
        expected = fuse_results([A, B], "combmnz")
        outcomes = []

        def work():
            for _ in range(200):
                outcomes.append(fuse_results([A, B], "combmnz") == expected)

        threads = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertTrue(all(outcomes) and len(outcomes) == 1600)


if __name__ == "__main__":
    unittest.main()
