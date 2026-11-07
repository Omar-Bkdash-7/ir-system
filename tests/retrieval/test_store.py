import threading
import time
import unittest

from common.exceptions import IndexNotFoundError, InvalidQueryError
from services.retrieval.core.compact import CompactCache
from services.retrieval.core.store import IndexStore
from tests.retrieval.helpers import TOY_DOCS, project_dir, write_bm25, write_embeddings


class FakeIndex:
    def __init__(self, dataset, kind, mb=1):
        self.dataset, self.kind, self.nbytes = dataset, kind, int(mb * 1024 * 1024)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class StoreFixture(unittest.TestCase):
    def setUp(self):
        self._ctx = project_dir()
        self.root = self._ctx.__enter__()
        self.addCleanup(self._ctx.__exit__, None, None, None)
        self.clock = Clock()
        self.loads = []
        self.gate = None
        self.delay = 0.0

    def files_of(self, kind, dataset):
        return [self.root / f"{dataset}.{kind}.bin"]

    def touch(self, dataset="a", kind="bm25", content=b"x"):
        (self.root / f"{dataset}.{kind}.bin").write_bytes(content)

    def loader(self, kind, dataset, **_):
        self.loads.append((dataset, kind))
        if self.gate is not None:
            self.gate.wait(5)
        if self.delay:
            time.sleep(self.delay)
        return FakeIndex(dataset, kind, mb=self.mb if hasattr(self, "mb") else 1)

    def store(self, **kwargs):
        defaults = dict(cache=CompactCache(None), stale_check_seconds=2.0, memory_budget_mb=0, loader=self.loader,
                        files_of=self.files_of, clock=self.clock)
        defaults.update(kwargs)
        return IndexStore(**defaults)


class LoadingTests(StoreFixture):
    def test_concurrent_first_requests_load_the_index_once(self):
        self.touch()
        self.delay = 0.2
        store = self.store()
        seen = []
        threads = [threading.Thread(target=lambda: seen.append(store.get("a", "bm25"))) for _ in range(16)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(len({id(i) for i in seen}), 1)
        self.assertEqual(len(seen), 16)

    def test_unknown_model_is_a_bad_request(self):
        with self.assertRaises(InvalidQueryError):
            self.store().get("a", "splade")

    def test_missing_index_is_not_found_and_nothing_is_cached(self):
        store = self.store()
        with self.assertRaises(IndexNotFoundError):
            store.get("a", "bm25")
        self.touch()
        self.assertIsNotNone(store.get("a", "bm25"))

    def test_failed_load_is_retried_next_time(self):
        self.touch()
        attempts = []

        def flaky(kind, dataset, **_):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("boom")
            return FakeIndex(dataset, kind)

        store = self.store(loader=flaky)
        with self.assertRaises(RuntimeError):
            store.get("a", "bm25")
        self.assertIsNotNone(store.get("a", "bm25"))
        self.assertEqual(len(attempts), 2)

    def test_dataset_ids_that_share_a_slug_share_an_index(self):
        self.touch("beir_quora")
        store = self.store(files_of=lambda kind, dataset: [self.root / "beir_quora.bm25.bin"])
        self.assertIs(store.get("beir/quora", "bm25"), store.get("beir-quora", "bm25"))
        self.assertEqual(len(self.loads), 1)


class FreshnessTests(StoreFixture):
    def test_files_are_not_checked_more_often_than_the_interval(self):
        self.touch()
        store = self.store()
        first = store.get("a", "bm25")
        (self.root / "a.bm25.bin").unlink()
        self.assertIs(store.get("a", "bm25"), first)  # inside the interval: no disk access
        self.clock.now += 5
        with self.assertRaises(IndexNotFoundError):
            store.get("a", "bm25")
        self.assertEqual(store.loaded(), [])  # the deleted index was released

    def test_changed_files_trigger_a_reload(self):
        self.touch(content=b"x")
        store = self.store()
        first = store.get("a", "bm25")
        self.clock.now += 5
        self.assertIs(store.get("a", "bm25"), first)  # unchanged: same object, no reload
        self.touch(content=b"a different, longer content")
        self.clock.now += 5
        second = store.get("a", "bm25")
        self.assertIsNot(second, first)
        self.assertEqual(len(self.loads), 2)

    def test_requests_are_served_from_memory_while_a_rebuilt_index_reloads(self):
        self.touch(content=b"x")
        store = self.store()
        old = store.get("a", "bm25")
        self.touch(content=b"changed")
        self.clock.now += 5
        self.gate = threading.Event()
        reloader = threading.Thread(target=lambda: store.get("a", "bm25"))
        reloader.start()
        time.sleep(0.1)  # the reload is now blocked inside the loader
        started = time.perf_counter()
        served = store.get("a", "bm25")
        self.assertLess(time.perf_counter() - started, 0.5)  # did not wait for the reload
        self.assertIs(served, old)
        self.gate.set()
        reloader.join()
        self.assertIsNot(store.get("a", "bm25"), old)


class MemoryBudgetTests(StoreFixture):
    def test_least_recently_used_index_is_released(self):
        for name in "ab":
            self.touch(name)
        self.mb = 2
        store = self.store(memory_budget_mb=3)
        a = store.get("a", "bm25")
        store.get("b", "bm25")  # 4 MB loaded > 3 MB budget: "a" (older) goes
        self.assertEqual([item["dataset"] for item in store.loaded()], ["b"])
        self.assertIsNotNone(a)  # whoever still holds a reference keeps a working object
        store.get("a", "bm25")
        self.assertEqual([item["dataset"] for item in store.loaded()], ["a"])
        self.assertEqual(self.loads.count(("a", "bm25")), 2)

    def test_the_index_just_loaded_is_never_released(self):
        self.touch("a")
        self.mb = 10
        store = self.store(memory_budget_mb=1)
        store.get("a", "bm25")
        self.assertEqual(len(store.loaded()), 1)


class DiscoveryTests(unittest.TestCase):
    def test_known_datasets_and_is_built(self):
        with project_dir() as root:
            write_bm25(root, "toy", TOY_DOCS)
            write_embeddings(root, "other", TOY_DOCS)
            store = IndexStore(cache=CompactCache(None))
            self.assertEqual(store.known_datasets(), ["other", "toy"])
            self.assertTrue(store.is_built("toy", "bm25"))
            self.assertFalse(store.is_built("toy", "embedding"))
            self.assertTrue(store.is_built("other", "embedding"))
            self.assertFalse(store.is_built("nothing", "bm25"))


if __name__ == "__main__":
    unittest.main()
