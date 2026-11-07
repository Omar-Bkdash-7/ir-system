import unittest

import numpy as np

from services.retrieval.core import kernels


class TopKTests(unittest.TestCase):
    def test_ties_are_broken_by_row_index(self):
        rows, scores = kernels.top_k(np.array([0, 3, 3, 1, 3, 0, 2], dtype=float), 2, positive_only=True)
        self.assertEqual(rows.tolist(), [1, 2])
        self.assertEqual(scores.tolist(), [3.0, 3.0])

    def test_matches_a_full_sort_on_random_input(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n, k = int(rng.integers(1, 80)), int(rng.integers(1, 90))
            values = rng.integers(0, 6, n).astype(float)
            order = sorted(range(n), key=lambda i: (-values[i], i))
            self.assertEqual(kernels.top_k(values, k)[0].tolist(), order[:k])
            self.assertEqual(kernels.top_k(values, k, positive_only=True)[0].tolist(), [i for i in order if values[i] > 0][:k])

    def test_empty_and_degenerate_inputs(self):
        self.assertEqual(kernels.top_k(np.zeros(5), 3, positive_only=True)[0].size, 0)
        self.assertEqual(kernels.top_k(np.zeros(0), 3)[0].size, 0)
        self.assertEqual(kernels.top_k(np.ones(3), 0)[0].size, 0)

    def test_negative_scores_are_kept_without_positive_only(self):
        rows, scores = kernels.top_k(np.array([-0.5, -0.1, -0.9]), 2)
        self.assertEqual(rows.tolist(), [1, 0])
        self.assertEqual(scores.dtype, np.float64)


class AccumulateTests(unittest.TestCase):
    def setUp(self):
        # 3 terms over 4 documents:   t0 -> d0(tf2) d2(tf1) | t1 -> d1(tf3) | t2 -> (no postings)
        self.offsets = np.array([0, 2, 3, 3], dtype=np.int64)
        self.doc_idx = np.array([0, 2, 1], dtype=np.int32)
        self.tf = np.array([2, 1, 3], dtype=np.uint8)
        self.idf = np.array([1.0, 2.0, 5.0])
        self.doc_len = np.array([10.0, 20.0, 30.0, 40.0])

    def test_bm25_scores_against_the_formula(self):
        k1, b, avg = 1.5, 0.75, 25.0
        scores = kernels.bm25_scores(self.offsets, self.doc_idx, self.tf, self.idf, self.doc_len, avg,
                                     np.array([0, 1, 2]), np.array([1.0, 2.0, 1.0]), k1, b)
        def part(idf, tf, dl):
            return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))
        expected = [part(1.0, 2, 10.0), 2 * part(2.0, 3, 20.0), part(1.0, 1, 30.0), 0.0]
        np.testing.assert_allclose(scores, expected, rtol=1e-12)

    def test_no_matching_postings_gives_all_zeros(self):
        scores = kernels.bm25_scores(self.offsets, self.doc_idx, self.tf, self.idf, self.doc_len, 25.0,
                                     np.array([2]), np.array([1.0]), 1.5, 0.75)
        self.assertEqual(scores.tolist(), [0.0] * 4)

    def test_sparse_dot_scores(self):
        values = np.array([0.5, 0.25, 1.0])
        scores = kernels.sparse_dot_scores(self.offsets, self.doc_idx, values, np.array([0, 1]), np.array([2.0, 0.5]), 4)
        np.testing.assert_allclose(scores, [1.0, 0.5, 0.5, 0.0])

    def test_gather_scores_zero_for_unknown_rows(self):
        out = kernels.gather_scores(np.array([1.0, 2.0, 3.0]), np.array([2, -1, 0]))
        self.assertEqual(out.tolist(), [3.0, 0.0, 1.0])

    def test_dense_scores(self):
        m = np.eye(3, dtype=np.float32)
        self.assertEqual(kernels.dense_scores(m, np.array([0.0, 2.0, 0.0], dtype=np.float32)).tolist(), [0.0, 2.0, 0.0])


if __name__ == "__main__":
    unittest.main()
