"""Loading queries and qrels from ir_datasets.

Loading is the slowest part of a run after retrieval itself, so each dataset is
read once per process and reused (a compare-all-models run loads it once, not
once per model). A dataset without qrels cannot be evaluated, and says so with a
``ValueError`` the HTTP layer turns into a 400.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from services.evaluation.core.dataset import DatasetSplit

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Read queries and relevance judgements for a dataset name."""

    def __init__(self, cache_size: int = 4) -> None:
        self._cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[str, DatasetSplit]" = OrderedDict()

    def load(self, dataset_name: str) -> DatasetSplit:
        """Queries and qrels of ``dataset_name`` (cached)."""
        name = (dataset_name or "").strip()
        if not name:
            raise ValueError("dataset_name is required")

        cached = self._cache.get(name)
        if cached is not None:
            self._cache.move_to_end(name)
            return cached

        split = self._read(name)
        self._cache[name] = split
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return split

    def validate_dataset(self, dataset_name: str) -> bool:
        """True when the dataset exists and has both queries and qrels."""
        try:
            self.load(dataset_name)
            return True
        except Exception as exc:  # noqa: BLE001 - a bad dataset name is a user error, not a crash
            logger.warning("Dataset %r is not evaluable: %s", dataset_name, exc)
            return False

    def load_queries_and_qrels(
        self, dataset_name: str
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
        """Backwards-compatible tuple view of :meth:`load`."""
        split = self.load(dataset_name)
        return split.queries, split.qrels

    def clear_cache(self) -> None:
        self._cache.clear()

    # --------------------------------------------------------------------------------------------- internals
    @staticmethod
    def _read(name: str) -> DatasetSplit:
        import ir_datasets  # imported lazily: unit tests do not need the dataset package

        logger.info("Loading dataset %s", name)
        try:
            dataset = ir_datasets.load(name)
        except KeyError as exc:
            raise ValueError(f"Unknown dataset: {name}") from exc

        queries = {query.query_id: query.text for query in dataset.queries_iter()}
        if not queries:
            raise ValueError(f"Dataset {name} has no queries")

        if not hasattr(dataset, "qrels_iter"):
            raise ValueError(f"Dataset {name} does not have qrels. Cannot evaluate.")

        qrels: Dict[str, Dict[str, int]] = {}
        for qrel in dataset.qrels_iter():
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)

        if not qrels:
            raise ValueError(f"Dataset {name} has empty qrels. Cannot evaluate.")

        logger.info("Loaded %d queries and %d qrels from %s", len(queries), len(qrels), name)
        return DatasetSplit(name=name, queries=queries, qrels=qrels)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DatasetLoader(cached={list(self._cache)})"
