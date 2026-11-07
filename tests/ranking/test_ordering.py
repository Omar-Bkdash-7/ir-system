import unittest

import numpy as np

from services.ranking.rankers import rank_by_cosine, rank_by_score, stable_order


def items(*scores):
    return [{"doc_id": f"d{i}", "score": s, "text": f"t{i}"} for i, s in enumerate(scores)]


class RankByScoreTests(unittest.TestCase):
    def test_orders_best_first(self):
        ranked = rank_by_score(items(0.2, 0.9, 0.5))
        self.assertEqual([r["doc_id"] for r in ranked], ["d1", "d2", "d0"])

    def test_equal_scores_keep_their_original_order(self):
        ranked = rank_by_score(items(1.0, 2.0, 1.0, 2.0, 1.0))
        self.assertEqual([r["doc_id"] for r in ranked], ["d1", "d3", "d0", "d2", "d4"])

    def test_top_k(self):
        data = items(5, 4, 3, 2, 1)
        self.assertEqual(len(rank_by_score(data, 2)), 2)
        self.assertEqual(len(rank_by_score(data, 99)), 5)
        self.assertEqual(rank_by_score(data, 0), [])
        self.assertEqual(len(rank_by_score(data, None)), 5)

    def test_negative_top_k_is_an_error_not_a_silent_truncation(self):
        with self.assertRaises(ValueError):
            rank_by_score(items(1, 2, 3), -1)

    def test_items_are_returned_untouched(self):
        data = items(0.1, 0.7)
        ranked = rank_by_score(data)
        self.assertIs(ranked[0], data[1])
        self.assertEqual(ranked[0], {"doc_id": "d1", "score": 0.7, "text": "t1"})

    def test_empty_input(self):
        self.assertEqual(rank_by_score([]), [])
        self.assertEqual(rank_by_score([], 5), [])

    def test_negative_and_integer_scores(self):
        self.assertEqual([r["score"] for r in rank_by_score(items(-1, -3, 2, 0))], [2, 0, -1, -3])

    def test_the_legacy_name_still_works(self):
        self.assertIs(rank_by_cosine, rank_by_score)

    def test_an_item_without_a_score_raises_key_error(self):
        with self.assertRaises(KeyError):
            rank_by_score([{"doc_id": "x"}])

    def test_stable_order_helper(self):
        self.assertEqual(stable_order(np.array([1.0, 3.0, 3.0, 2.0])).tolist(), [1, 2, 3, 0])


if __name__ == "__main__":
    unittest.main()
