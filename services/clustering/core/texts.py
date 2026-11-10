"""Document texts for topic detection.

Topic quality depends entirely on having the real text: the previous implementation fell back to "Document <id>"
placeholders, which is why the stored topics were built from the word "document" and empty strings. Here a failed
lookup is reported (TextsUnavailableError / missing ratio) and the caller disables topics instead of inventing text.

Texts live in PostgreSQL, one table per dataset (dataset_<slug>: doc_id, text) - the same layout the indexing
service writes and the retrieval service reads, so the reader is reused instead of copied.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from services.clustering.errors import TextsUnavailableError

logger = logging.getLogger(__name__)

#: A lookup function (dataset, doc_ids) -> {doc_id: text}. Injected in tests.
Fetcher = Callable[[str, Sequence[str]], Dict[str, str]]


@dataclass(frozen=True)
class DocumentTexts:
    """Texts aligned with the embedding index's doc_ids, plus how much of the dataset is missing."""

    texts: List[str]
    missing: int

    @property
    def total(self) -> int:
        return len(self.texts)

    @property
    def missing_ratio(self) -> float:
        return (self.missing / self.total) if self.total else 0.0

    def usable(self, max_missing_ratio: float) -> bool:
        """Enough real text to trust keyword extraction (empty texts carry no topic signal)."""
        return self.total > 0 and self.missing_ratio <= max_missing_ratio


def _resolve_fetcher(store: Optional[object]) -> Callable[[str, Sequence[str]], Dict[str, str]]:
    if store is None:
        from services.retrieval.core.texts import TextStore

        store = TextStore()
    return store.fetch


def fetch_texts(
    dataset: str,
    doc_ids: Sequence[str],
    *,
    batch_size: int = 1000,
    store: Optional[object] = None,
    fetcher: Optional[Fetcher] = None,
) -> DocumentTexts:
    """{doc_id: text} for this dataset, in doc_ids order.

    Raises TextsUnavailableError when the database cannot be queried at all (the caller then reports that topics
    are unavailable rather than failing the clustering request, which only needs the vectors).
    """
    if not doc_ids:
        return DocumentTexts(texts=[], missing=0)
    fetch = fetcher or _resolve_fetcher(store)

    batch_size = max(1, int(batch_size))
    found: Dict[str, str] = {}
    try:
        for start in range(0, len(doc_ids), batch_size):
            batch = [str(doc_id) for doc_id in doc_ids[start:start + batch_size]]
            found.update(fetch(dataset, batch))
    except TextsUnavailableError:
        raise
    except Exception as exc:  # connection refused, missing table, missing psycopg2...
        raise TextsUnavailableError(f"Could not read document texts for dataset {dataset!r}: {exc}") from exc

    texts: List[str] = []
    missing = 0
    for doc_id in doc_ids:
        text = found.get(str(doc_id)) or ""
        if not text.strip():
            missing += 1
        texts.append(text)

    if missing:
        logger.warning("%d of %d documents have no text (dataset=%s)", missing, len(texts), dataset)
    return DocumentTexts(texts=texts, missing=missing)
