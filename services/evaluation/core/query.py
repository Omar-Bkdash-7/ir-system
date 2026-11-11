"""Turning a dataset query into what a retriever expects.

Evaluation must score exactly what search would score, so this is the one place
that decides how a query is refined and tokenised. Both steps degrade instead of
failing: a refinement outage means "use the original query" (a warning is logged
once), and a preprocessing outage means "split on whitespace".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .clients import JsonClient, UpstreamServiceError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedQuery:
    """Query text as used for retrieval, plus the tokens the lexical models want."""

    text: str
    tokens: Tuple[str, ...]
    refined: bool = False


class QueryPipeline:
    """Refine (optional) then tokenise a query through the shared services."""

    def __init__(
        self,
        client: JsonClient,
        *,
        preprocessing_url: str,
        refinement_url: str,
        preprocessing_timeout: float = 10.0,
        refinement_timeout: float = 10.0,
    ) -> None:
        self._client = client
        self._preprocessing_url = preprocessing_url.rstrip("/")
        self._refinement_url = refinement_url.rstrip("/")
        self._preprocessing_timeout = preprocessing_timeout
        self._refinement_timeout = refinement_timeout
        self._preprocessing_warned = False
        self._refinement_warned = False

    def prepare(self, text: str, *, dataset: Optional[str] = None, refine: bool = False) -> PreparedQuery:
        """The query to retrieve with, and its tokens."""
        query = self.refine(text, dataset) if refine else text
        return PreparedQuery(text=query, tokens=self.tokens(query), refined=query != text)

    def refine(self, text: str, dataset: Optional[str] = None) -> str:
        """Spelling-only refinement (evaluation mode) so runs stay deterministic."""
        try:
            body = self._client.post(
                f"{self._refinement_url}/refine",
                {
                    "query": text,
                    "history": [],
                    "enabled": True,
                    "is_evaluation": True,
                    "dataset": dataset,
                },
                timeout=self._refinement_timeout,
                service="query_refinement",
            )
        except UpstreamServiceError as exc:
            if not self._refinement_warned:
                self._refinement_warned = True
                logger.warning("Query refinement unavailable (%s); evaluating original queries", exc)
            return text

        refined = str(body.get("refined_query") or text)
        if refined != text:
            logger.debug("Query refined: %r -> %r", text, refined)
        return refined

    def tokens(self, text: str) -> Tuple[str, ...]:
        """Tokens from the preprocessing service, falling back to a simple split."""
        try:
            body = self._client.post(
                f"{self._preprocessing_url}/preprocess",
                {"text": text, "mode": "query"},
                timeout=self._preprocessing_timeout,
                service="preprocessing",
            )
        except UpstreamServiceError as exc:
            if not self._preprocessing_warned:
                self._preprocessing_warned = True
                logger.warning("Preprocessing unavailable (%s); tokenising on whitespace", exc)
            return self._split(text)

        tokens = body.get("tokens")
        if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)):
            return tuple(str(token) for token in tokens if str(token).strip())
        return self._split(text)

    @staticmethod
    def _split(text: str) -> Tuple[str, ...]:
        return tuple(text.lower().split())
