"""Old fusion / ordering code vs the new engine on thousands of random inputs: the results must be IDENTICAL
(same documents, same float scores bit for bit, same order including ties, same errors).

tests/ranking/legacy holds the code exactly as it was before the rewrite. Cases cover 0..6 lists, empty and one-item lists,
duplicates inside a list, integer / negative / all-equal / tiny / unsorted scores, every method (any letter case), valid and
invalid weights, and several rrf_k values.   RANKING_TEST_CASES=20000 for more cases.
"""
import logging
import os
import random
import unittest

from services.ranking.rankers import (combmnz_fusion, combsum_fusion, fuse_results, normalize_scores, rank_by_cosine,
                                      reciprocal_rank_fusion, weighted_fusion)
from tests.ranking.legacy import cosine_ranker as old_ordering
from tests.ranking.legacy import score_fusion as old_fusion

CASES = int(os.getenv("RANKING_TEST_CASES", "2500"))


def outcome(function, *args, **kwargs):
    try:
        return ("ok", function(*args, **kwargs))
    except Exception as error:  # the type and message of the error are part of the contract
        return ("error", type(error).__name__, str(error))


class Generator:
    MODES = ("float", "int", "equal", "negative", "unsorted", "tiny", "unit")

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def hit_list(self, pool, size, mode, duplicates):
        rng = self.rng
        ids = [rng.choice(pool) for _ in range(size)] if duplicates else rng.sample(pool, min(size, len(pool)))
        if mode == "float":
            scores = sorted((rng.uniform(-5, 20) for _ in ids), reverse=True)
        elif mode == "int":
            scores = sorted((rng.randint(0, 4) for _ in ids), reverse=True)  # lots of ties
        elif mode == "equal":
            scores = [2.5] * len(ids)
        elif mode == "negative":
            scores = sorted((-rng.random() * 3 for _ in ids), reverse=True)
        elif mode == "unsorted":
            scores = [rng.uniform(0, 1) for _ in ids]
        elif mode == "tiny":
            scores = sorted((rng.random() * 1e-12 for _ in ids), reverse=True)
        else:
            scores = sorted((rng.random() for _ in ids), reverse=True)
        return [{"doc_id": doc_id, "score": score} for doc_id, score in zip(ids, scores)]

    def lists(self):
        rng = self.rng
        pool = [f"d{i}" for i in range(rng.choice([3, 10, 60, 400]))]
        return [self.hit_list(pool, rng.choice([0, 1, 2, 3, 8, 30, 150]), rng.choice(self.MODES), rng.random() < 0.1)
                for _ in range(rng.choice([0, 1, 2, 2, 3, 3, 4, 6]))]


class DifferentialTests(unittest.TestCase):
    def setUp(self):
        logging.getLogger("services.ranking.rankers.score_fusion").setLevel(logging.ERROR)
        logging.getLogger("tests.ranking.legacy.score_fusion").setLevel(logging.ERROR)
        logging.getLogger("tests.ranking.legacy.cosine_ranker").setLevel(logging.ERROR)

    def test_fuse_results_is_identical_to_the_old_code(self):
        gen = Generator(20240607)
        rng = gen.rng
        errors = 0
        for case in range(CASES):
            lists = gen.lists()
            method = rng.choice(["rrf", "RRF", "combsum", "combmnz", "weighted", "Weighted", "bogus"])
            k = rng.choice([0, 1, 5, 60, 100])
            weights = rng.choice([None, [], [rng.random() for _ in lists], [rng.random() for _ in range(len(lists) + 1)]])
            expected = outcome(old_fusion.fuse_results, lists, method=method, weights=weights, rrf_k=k)
            actual = outcome(fuse_results, lists, method=method, weights=weights, rrf_k=k)
            self.assertEqual(actual, expected, f"case {case}: method={method} k={k} weights={weights}")
            if expected[0] == "ok" and len(lists) == 1:
                self.assertIs(actual[1], lists[0])
            errors += expected[0] == "error"
        self.assertGreater(errors, 0)  # the error paths were exercised too

    def test_the_named_fusion_functions_are_identical(self):
        gen = Generator(7)
        for case in range(CASES):
            lists = gen.lists()
            for name, old, new in (("rrf", old_fusion.reciprocal_rank_fusion, reciprocal_rank_fusion),
                                   ("combsum", old_fusion.combsum_fusion, combsum_fusion),
                                   ("combmnz", old_fusion.combmnz_fusion, combmnz_fusion)):
                self.assertEqual(outcome(new, lists), outcome(old, lists), f"case {case} {name}")
            weights = [gen.rng.random() for _ in lists]
            self.assertEqual(outcome(weighted_fusion, lists, weights), outcome(old_fusion.weighted_fusion, lists, weights), f"case {case} weighted")
            self.assertEqual(outcome(weighted_fusion, lists, weights + [1.0]), outcome(old_fusion.weighted_fusion, lists, weights + [1.0]))
            for hits in lists:
                self.assertEqual(outcome(normalize_scores, hits), outcome(old_fusion.normalize_scores, hits), f"case {case} normalize")

    def test_top_k_is_a_prefix_of_the_full_result(self):
        gen = Generator(11)
        for _ in range(CASES // 2):
            lists = gen.lists()
            if len(lists) < 2:
                continue
            method = gen.rng.choice(["rrf", "combsum", "combmnz"])
            full = fuse_results(lists, method)
            for top_k in (1, 3, 10, 10_000):
                self.assertEqual(fuse_results(lists, method, top_k=top_k), full[:top_k])

    def test_ordering_is_identical_to_the_old_rank_by_cosine(self):
        rng = random.Random(3)
        for case in range(CASES):
            items = [{"doc_id": f"x{i}", "score": rng.choice([rng.randint(0, 5), rng.random() * 3, -rng.random()]), "text": "t" * rng.randint(0, 3)}
                     for i in range(rng.choice([0, 1, 2, 7, 60]))]
            top_k = rng.choice([None, 0, 1, 5, 100])
            expected = outcome(old_ordering.rank_by_cosine, items, top_k)
            actual = outcome(rank_by_cosine, items, top_k)
            self.assertEqual(actual, expected, f"case {case} top_k={top_k}")
            if expected[0] == "ok":
                self.assertTrue(all(a is b for a, b in zip(actual[1], expected[1])))


if __name__ == "__main__":
    unittest.main()
