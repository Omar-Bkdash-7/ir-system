"""Old vs new ranking code, no data needed. Run from the project root:   python -m tests.ranking.bench_fusion

'old' is the pure-Python code this rewrite replaced (tests/ranking/legacy); 'new' is services.ranking.rankers. The sizes are
the ones the retrieval service really sends to the fusion: up to 3 lists of 1,000 candidates (the default), or 2 lists of 5,000
(the maximum). Medians of repeated calls, in microseconds; results are checked to be identical before they are timed.
"""
import logging
import random
import statistics
import time

from services.ranking.rankers import fuse_results, rank_by_cosine
from tests.ranking.legacy import cosine_ranker as old_ordering
from tests.ranking.legacy import score_fusion as old_fusion

for name in ("services.ranking.rankers.score_fusion", "tests.ranking.legacy.score_fusion"):
    logging.getLogger(name).setLevel(logging.ERROR)


def make_lists(n_lists, size, overlap_pool, seed=1):
    rng = random.Random(seed)
    return [[{"doc_id": f"D{rng.randrange(overlap_pool)}", "score": s} for s in sorted((rng.random() * 20 for _ in range(size)), reverse=True)]
            for _ in range(n_lists)]


def median_us(function, repeats):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append((time.perf_counter() - started) * 1e6)
    return statistics.median(times)


def main():
    print(f"{'case':<34}{'old (us)':>12}{'new (us)':>12}{'speedup':>10}")
    for label, n_lists, size, pool, repeats in (("3 lists x 100 docs", 3, 100, 300, 400), ("3 lists x 1000 docs (default)", 3, 1000, 3000, 200),
                                                ("2 lists x 5000 docs (maximum)", 2, 5000, 20000, 40)):
        lists = make_lists(n_lists, size, pool)
        weights = [1.0 / n_lists] * n_lists
        for method in ("rrf", "combsum", "combmnz", "weighted"):
            args = {"method": method, "weights": weights if method == "weighted" else None, "rrf_k": 60}
            assert old_fusion.fuse_results(lists, **args) == fuse_results(lists, **args), "results differ"
            old_us = median_us(lambda: old_fusion.fuse_results(lists, **args), repeats)
            new_us = median_us(lambda: fuse_results(lists, **args), repeats)
            print(f"{label + ' ' + method:<34}{old_us:>12.0f}{new_us:>12.0f}{old_us / new_us:>9.1f}x")
    for count, repeats in ((1000, 300), (10000, 40)):
        items = [{"doc_id": f"D{i}", "score": random.Random(i).random(), "text": "t"} for i in range(count)]
        assert old_ordering.rank_by_cosine(items, 50) == rank_by_cosine(items, 50)
        old_us = median_us(lambda: old_ordering.rank_by_cosine(items, 50), repeats)
        new_us = median_us(lambda: rank_by_cosine(items, 50), repeats)
        print(f"{'order ' + str(count) + ' items, top 50':<34}{old_us:>12.0f}{new_us:>12.0f}{old_us / new_us:>9.1f}x")


if __name__ == "__main__":
    main()
