"""Model names as the API, the storage and the frontend spell them."""
from __future__ import annotations

SINGLE_MODELS = ("tfidf", "bm25", "embedding")
HYBRID_MODELS = ("hybrid_serial", "hybrid_parallel")
SUPPORTED_MODELS = SINGLE_MODELS + HYBRID_MODELS

FUSION_METHODS = ("rrf", "weighted", "combsum", "combmnz")
#: What /status reports as the metrics the service computes.
METRIC_NAMES = ("MAP", "nDCG", "Recall", "Precision")

#: Human label of a model, used in stored rows and in /metrics keys.
LABELS = {
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "embedding": "EMBEDDING",
    "hybrid_serial": "HYBRID-SERIAL",
    "hybrid_parallel": "HYBRID-PARALLEL",
}


def require_model(model: str | None) -> str:
    """Normalise a model name or reject it before any work is done."""
    name = (model or "").strip().lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Model must be one of {list(SUPPORTED_MODELS)}, got {model!r}")
    return name


def require_fusion(method: str | None) -> str:
    """Normalise a fusion method; ``None`` means the RRF default."""
    name = (method or "rrf").strip().lower()
    if name not in FUSION_METHODS:
        raise ValueError(f"Fusion method must be one of {list(FUSION_METHODS)}, got {method!r}")
    return name


def is_hybrid(model: str) -> bool:
    return model in HYBRID_MODELS


def display_label(model: str) -> str:
    """Key used by /metrics for a model: ``tfidf`` -> ``TF-IDF``."""
    return LABELS.get(model, (model or "").upper().replace("_", "-"))
