"""storage.py (atomic writes, build locks) and the loaded-index cache."""
import os
import time
import unittest

from services.indexing.services import index_cache
from services.indexing.storage import atomic_path, try_lock
from tests.indexing._support import WorkdirTestCase


class AtomicPathTest(WorkdirTestCase):
    def test_success_replaces_the_file_and_leaves_no_temp_file(self):
        target = self.work / "idx" / "file.bin"
        with atomic_path(target) as tmp:
            self.assertNotEqual(tmp, target)
            tmp.write_bytes(b"new")
        self.assertEqual(target.read_bytes(), b"new")
        self.assertEqual([p.name for p in target.parent.iterdir()], ["file.bin"])

    def test_a_failure_keeps_the_previous_file_untouched(self):
        target = self.work / "file.bin"
        target.write_bytes(b"old")
        with self.assertRaises(RuntimeError):
            with atomic_path(target) as tmp:
                tmp.write_bytes(b"half written")
                raise RuntimeError("crash while saving")
        self.assertEqual(target.read_bytes(), b"old")
        self.assertEqual([p.name for p in self.work.iterdir() if p.is_file()], ["file.bin"])


class TryLockTest(unittest.TestCase):
    def test_second_holder_is_refused_until_the_first_is_done(self):
        with try_lock("test:a") as first:
            self.assertTrue(first)
            with try_lock("test:a") as second:
                self.assertFalse(second)
            with try_lock("test:b") as other_key:
                self.assertTrue(other_key)
        with try_lock("test:a") as again:
            self.assertTrue(again)

    def test_released_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with try_lock("test:c"):
                raise RuntimeError
        with try_lock("test:c") as acquired:
            self.assertTrue(acquired)


class IndexCacheTest(WorkdirTestCase):
    def _file(self, name, content="x"):
        path = self.work / name
        path.write_text(content)
        return path

    def test_unchanged_files_are_loaded_once(self):
        f = self._file("a.pkl")
        calls = []
        factory = lambda: calls.append(1) or object()  # noqa: E731
        first = index_cache.load_cached(factory, [f])
        self.assertIs(index_cache.load_cached(factory, [f]), first)
        self.assertEqual(len(calls), 1)

    def test_a_rewritten_file_is_noticed(self):
        f = self._file("a.pkl", "one")
        calls = []
        factory = lambda: calls.append(1) or object()  # noqa: E731
        first = index_cache.load_cached(factory, [f])
        time.sleep(0.01)
        f.write_text("different and longer content")
        os.utime(f, (time.time() + 5, time.time() + 5))
        self.assertIsNot(index_cache.load_cached(factory, [f]), first)
        self.assertEqual(len(calls), 2)

    def test_only_a_few_indexes_are_kept_in_memory(self):
        files = [self._file(f"f{i}.pkl", str(i)) for i in range(4)]
        for f in files:
            index_cache.load_cached(lambda: object(), [f])
        calls = []
        index_cache.load_cached(lambda: calls.append(1) or object(), [files[0]])  # evicted long ago
        self.assertEqual(len(calls), 1)

    def test_clear(self):
        f = self._file("a.pkl")
        calls = []
        factory = lambda: calls.append(1) or object()  # noqa: E731
        index_cache.load_cached(factory, [f])
        index_cache.clear_index_cache()
        index_cache.load_cached(factory, [f])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
