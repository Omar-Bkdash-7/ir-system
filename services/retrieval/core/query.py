"""ONE policy for turning a request's query into something each model can use.

BM25 / TF-IDF   query_tokens when the caller sent them (the gateway and the evaluation service already ran the
                preprocessing service on the query); otherwise query_text is preprocessed here, once. Tokens that were
                already preprocessed are NEVER preprocessed a second time.
Embedding       query_text is required. embedding_query_mode "preprocessed" embeds the preprocessed tokens joined by
                spaces (what the service always did); "raw" embeds the original text (the indexer embeds each
                document's raw text).

The preprocessing package (spaCy) is imported lazily, only when a query really needs it.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from common.exceptions import InvalidQueryError

from .names import LABELS


def default_preprocess(text: str) -> List[str]:
    from services.preprocessing.preprocessing_service import preprocess  # loads spaCy: import only when needed

    return list(preprocess(text, mode="query"))


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
