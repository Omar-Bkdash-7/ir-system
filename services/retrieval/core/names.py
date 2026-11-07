"""Names shared by the core modules."""
from __future__ import annotations

KINDS = ("tfidf", "embedding", "bm25")
LABELS = {"tfidf": "TF-IDF", "embedding": "Embedding", "bm25": "BM25"}


def dataset_slug(dataset: str) -> str:
    """Directory slug of a dataset: the rule of common.config.get_dataset_specific_path ('/', '-' and '.' become '_')."""
    return dataset.replace("/", "_").replace("-", "_").replace(".", "_")
