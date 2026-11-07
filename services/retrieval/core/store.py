"""Loads each (dataset, model) index once, keeps it fresh, and keeps total memory bounded.

* one lock per index: concurrent first requests load it once, the others wait for that load (the old code loaded it
  several times and, once loaded, never looked at the files again);
* the index files are checked (mtime + size) at most every `stale_check_seconds`; a rebuilt index is reloaded, and while
  that happens requests keep being served by the index that is already in memory (stale-while-revalidate);
* an index whose files were deleted is dropped and answers 404;
* when the loaded indexes exceed the memory budget the least recently used ones are released (requests that are still
  using one keep working: they hold a reference).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from common.config import get_config
from common.exceptions import IndexNotFoundError, InvalidQueryError

from .compact import CompactCache
from .indexes import BaseIndex, load_index, source_files
from .names import KINDS, LABELS, dataset_slug

logger = logging.getLogger(__name__)

Key = Tuple[str, str]  # (dataset slug, kind)
Stamp = Tuple[Tuple[str, int, int], ...]


@dataclass
class _Entry:
    index: BaseIndex
    stamp: Stamp
    checked_at: float


class IndexStore:
    def __init__(
        self,
        *,
        cache: CompactCache,
        stale_check_seconds: float = 2.0,
        memory_budget_mb: int = 2048,
        embedding_mmap: bool = True,
        loader: Callable[..., BaseIndex] = load_index,
        files_of: Callable[[str, str], List[Path]] = source_files,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cache = cache
        self._stale_after = stale_check_seconds
        self._budget_bytes = int(memory_budget_mb) * 1024 * 1024  # 0 = unlimited
        self._mmap = embedding_mmap
        self._loader = loader
        self._files_of = files_of
        self._clock = clock
        self._entries: "OrderedDict[Key, _Entry]" = OrderedDict()
        self._locks: Dict[Key, threading.Lock] = {}
        self._guard = threading.Lock()  # protects _entries and _locks; never held while an index loads

    # ---------------------------------------------------------------------------------------------------------------
    def get(self, dataset: str, kind: str) -> BaseIndex:
        if kind not in KINDS:
            raise InvalidQueryError(f"Unknown retrieval model: {kind}")
        key = (dataset_slug(dataset), kind)
        with self._guard:
            entry = self._entries.get(key)
            if entry is not None and self._clock() - entry.checked_at < self._stale_after:
                self._entries.move_to_end(key)
                return entry.index
            lock = self._locks.setdefault(key, threading.Lock())
        return self._refresh(dataset, kind, key, lock, entry)

    def _refresh(self, dataset: str, kind: str, key: Key, lock: threading.Lock, entry: Optional[_Entry]) -> BaseIndex:
        if entry is not None:
            if not lock.acquire(blocking=False):
                return entry.index  # someone else is already reloading: keep serving what is in memory
        else:
            lock.acquire()
        try:
            try:
                stamp = self._stamp(kind, dataset)
            except IndexNotFoundError:
                self._drop(key)
                raise
            with self._guard:
                current = self._entries.get(key)
            if current is not None and current.stamp == stamp:
                current.checked_at = self._clock()
                return current.index
            if current is not None:
                logger.info("%s index of %s changed on disk: reloading", LABELS[kind], dataset)
            index = self._loader(kind, dataset, embedding_mmap=self._mmap, cache=self._cache)
            self._install(key, _Entry(index, stamp, self._clock()))
            return index
        finally:
            lock.release()

    def _stamp(self, kind: str, dataset: str) -> Stamp:
        stamp = []
        for path in self._files_of(kind, dataset):
            try:
                info = path.stat()
            except OSError:
                raise IndexNotFoundError(f"{LABELS[kind]} index not found at {path}") from None
            stamp.append((path.name, info.st_mtime_ns, info.st_size))
        return tuple(stamp)

    def _install(self, key: Key, entry: _Entry) -> None:
        with self._guard:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            if self._budget_bytes <= 0:
                return
            total = sum(item.index.nbytes for item in self._entries.values())
            for old_key in list(self._entries):  # least recently used first
                if total <= self._budget_bytes:
                    break
                if old_key == key:
                    continue
                released = self._entries.pop(old_key)
                total -= released.index.nbytes
                logger.warning("Released the %s index of %s (%.0f MB) to stay within the %d MB index memory budget",
                               LABELS[old_key[1]], old_key[0], released.index.nbytes / 1048576, self._budget_bytes // 1048576)

    def _drop(self, key: Key) -> None:
        with self._guard:
            if self._entries.pop(key, None) is not None:
                logger.warning("%s index of %s disappeared from disk: released", LABELS[key[1]], key[0])

    # ---------------------------------------------------------------------------------------------------------------
    def is_built(self, dataset: str, kind: str) -> bool:
        try:
            return all(path.exists() for path in self._files_of(kind, dataset))
        except ValueError:
            return False

    def known_datasets(self) -> List[str]:
        """Dataset slugs that have a directory below the configured index folders (nothing is loaded)."""
        roots = set()
        for key in ("indices.bm25_index", "indices.tfidf_matrix", "indices.embeddings"):
            template = get_config(key, None)
            if isinstance(template, str) and "{dataset}" in template:
                roots.add(Path(template.split("{dataset}")[0] or "."))
        if not roots:
            roots.add(Path(get_config("indices.base_dir", "data/indices")))
        names = set()
        for root in roots:
            try:
                names.update(entry.name for entry in root.iterdir() if entry.is_dir())
            except OSError:
                continue
        return sorted(names)

    def loaded(self) -> List[Dict[str, Any]]:
        with self._guard:
            return [
                {"dataset": slug, "model": kind, "memory_mb": round(entry.index.nbytes / 1048576, 2)}
                for (slug, kind), entry in self._entries.items()
            ]
