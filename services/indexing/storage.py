"""Crash-safe file writing and per-index build locks."""
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator


@contextmanager
def atomic_path(final: Path) -> Iterator[Path]:
    """Yield a temporary path next to `final`; on success rename it over `final` (atomic on one filesystem).
    A crash or an exception can therefore never leave a truncated index under its real name."""
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + ".tmp")
    try:
        yield tmp
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


_LOCKS: Dict[str, threading.Lock] = {}
_REGISTRY_GUARD = threading.Lock()


@contextmanager
def try_lock(key: str) -> Iterator[bool]:
    """Non-blocking named lock (the service runs ONE uvicorn worker). Yields False when `key` is already held,
    so a second identical build request is refused instead of corrupting the first one's files."""
    with _REGISTRY_GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    acquired = lock.acquire(blocking=False)
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()
