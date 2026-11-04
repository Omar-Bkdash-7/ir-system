"""Tiny LRU cache of LOADED indexers, invalidated by file modification time.

The status and inverted-index explorer endpoints used to unpickle the whole index on EVERY request (the explorer calls
them on each keystroke). The cache key contains (path, mtime, size) of every index file, so a rebuild or a delete is
noticed immediately and a stale index is never served.
"""
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, Tuple, TypeVar

T = TypeVar("T")

_MAX_ENTRIES = 2  # an index can be hundreds of MB in RAM: keep very few
_cache: "OrderedDict[Tuple, object]" = OrderedDict()
_lock = Lock()


def _stamp(path: Path) -> Tuple[str, int, int]:
    st = path.stat()
    return str(path), st.st_mtime_ns, st.st_size


def load_cached(factory: Callable[[], T], files: Iterable[Path]) -> T:
    """factory() must return a LOADED indexer. `files` are the index files whose state identifies the content."""
    key = tuple(_stamp(p) for p in files)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit  # type: ignore[return-value]
    value = factory()  # loading is slow: done outside the lock
    with _lock:
        _cache[key] = value
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)
    return value


def clear_index_cache() -> None:
    with _lock:
        _cache.clear()
