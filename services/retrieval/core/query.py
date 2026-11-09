"""ONE policy for turning a request's query into something each model can use.

BM25 / TF-IDF   query_tokens when the caller sent them (the gateway and the evaluation service already ran the
                preprocessing service on the query); otherwise query_text is preprocessed here, once. Tokens that were
                already preprocessed are NEVER preprocessed a second time.
Embedding       query_text is required. embedding_query_mode "raw" embeds the original text (the indexer embeds each
                document's raw text); "preprocessed" embeds the preprocessed tokens joined by spaces.

A text-only query is preprocessed with the configuration the dataset's index was built with (the metadata the
indexing service writes), not with the defaults: otherwise a dataset built with a custom config has its documents and
the queries that search them tokenised differently. The preprocessing package (spaCy) is imported lazily, only when a
query really needs it.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.config import get_config
from common.exceptions import InvalidQueryError

from .names import LABELS

logger = logging.getLogger(__name__)


def default_preprocess(text: str) -> List[str]:
    from services.preprocessing.preprocessing_service import preprocess  # loads spaCy: import only when needed

    return list(preprocess(text, mode="query"))


_saved_configs: Dict[Tuple[str, int, int], Optional[dict]] = {}
_saved_configs_lock = threading.Lock()
_SUPPORTED_CACHED = 64


def _metadata_path(dataset: str) -> Path:
    """The metadata file the indexing service writes for a dataset (same directory and naming rule)."""
    safe = str(dataset).replace("/", "-").replace("\\", "-")
    return Path(get_config("paths.metadata_dir", "data/metadata")) / f"{safe}.json"


def saved_preprocessing_config(dataset: Optional[str]) -> Optional[dict]:
    """The preprocessing config the dataset's indexes were built with, or None. Never raises.

    The file is stat'ed on every call (a few microseconds) so a rebuild is picked up without a restart, and parsed
    once per version of the file.
    """
    if not dataset or not str(dataset).strip():
        return None
    path = _metadata_path(str(dataset))
    try:
        info = path.stat()
    except OSError:
        return None  # no metadata: the dataset was never built with a custom config
    key = (str(path), info.st_mtime_ns, info.st_size)
    with _saved_configs_lock:
        if key in _saved_configs:
            return _saved_configs[key]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read the metadata of dataset %r (%s); using the default preprocessing", dataset, exc)
        return None
    config = metadata.get("preprocessing_config") if isinstance(metadata, dict) else None
    with _saved_configs_lock:
        if len(_saved_configs) >= _SUPPORTED_CACHED:
            _saved_configs.clear()
        _saved_configs[key] = config
    return config


def preprocessor_for(dataset: Optional[str]) -> Callable[[str], List[str]]:
    """`preprocess(text)` bound to the configuration the dataset's index was built with.

    A caller that sends query_tokens preprocessed the query itself (api_gateway does); a caller that sends only text
    (the RAG service) used to get the DEFAULTS here, so an index built with a custom config was searched with
    differently tokenised queries. An unreadable or unknown config falls back to the defaults.
    """
    config = saved_preprocessing_config(dataset)
    if not config:
        return default_preprocess

    def preprocess(text: str) -> List[str]:
        from services.preprocessing.preprocessing_service import PreprocessingConfig, preprocess as run

        try:
            return list(run(text, config=PreprocessingConfig.from_dict(config)))
        except (TypeError, ValueError) as exc:  # a config written by another version of the service
            logger.warning("Ignoring the saved preprocessing config of %r (%s); using the defaults", dataset, exc)
            return default_preprocess(text)

    return preprocess


class QueryContext:
    """One incoming query, prepared at most once per request and shared by every model of a hybrid request."""

    def __init__(
        self,
        *,
        text: Optional[str],
        tokens: Optional[Sequence[str]],
        encoder: Any,
        embedding_query_mode: str = "preprocessed",
        preprocess: Callable[[str], Sequence[str]] = default_preprocess,
    ) -> None:
        self.text = text or ""
        self._given: Tuple[str, ...] = tuple(token for token in (tokens or ()) if token and str(token).strip())
        self._encoder = encoder
        self._mode = embedding_query_mode
        self._preprocess = preprocess
        self._lock = threading.RLock()
        self._lexical: Optional[Tuple[str, ...]] = None
        self._dense_text: Optional[str] = None

    def require(self, kind: str) -> None:
        """Fail fast (400) when the query lacks what this model needs; does no work."""
        if kind == "embedding":
            if not self.text.strip():
                raise InvalidQueryError("query_text is required for embedding retrieval")
        elif not self._given and not self.text.strip():
            raise InvalidQueryError(f"query_tokens or query_text is required for {LABELS[kind]} retrieval")

    def lexical_tokens(self, kind: str) -> Tuple[str, ...]:
        with self._lock:
            if self._lexical is None:
                if self._given:
                    self._lexical = self._given
                else:
                    self.require(kind)
                    self._lexical = tuple(token for token in self._preprocess(self.text) if token and str(token).strip())
            return self._lexical

    def dense_text(self) -> str:
        with self._lock:
            if self._dense_text is None:
                self.require("embedding")
                if self._mode == "raw":
                    self._dense_text = self.text.strip()
                else:
                    self._dense_text = " ".join(self.lexical_tokens("embedding")) or self.text
            return self._dense_text

    def dense_vector(self) -> np.ndarray:
        return self._encoder.encode(self.dense_text())
