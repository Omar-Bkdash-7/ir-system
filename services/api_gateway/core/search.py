"""Search orchestration: refine the query (optional), tokenise it, call retrieval.

Framework-free on purpose: requests are read with ``getattr`` and the answers are
plain dicts, so the whole path can be tested without FastAPI, pydantic, a running
retrieval service or a dataset on disk.

Two behaviours are load-bearing and covered by tests:

* A query that preprocessing empties (``"who are you"``) is still sent to
  retrieval: the embedding model embeds ``query_text`` and never reads the tokens,
  so answering here with an empty result would hide the query from every model.
* The dataset's saved preprocessing config is read straight from the metadata
  file, never through ``services.indexing`` (which imports ``ir_datasets``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from common.http_client import JsonClient
from services.api_gateway.config import GatewaySettings, get_settings

from .dataset import dataset_info, preprocessing_config

logger = logging.getLogger(__name__)


class SearchService:
    """Everything the search routes need, in one place."""

    def __init__(
        self,
        settings: GatewaySettings,
        client: Optional[JsonClient] = None,
        *,
        metadata_dir: Optional[str] = None,
    ) -> None:
        self._settings = settings
        self._client = client or JsonClient(
            max_retries=settings.max_retries, backoff_seconds=settings.retry_backoff
        )
        self._metadata_dir = metadata_dir or settings.metadata_dir

    @classmethod
    def create(cls, settings: Optional[GatewaySettings] = None) -> "SearchService":
        return cls(settings or get_settings())

    @property
    def settings(self) -> GatewaySettings:
        return self._settings

    def close(self) -> None:
        self._client.close()

    # ----------------------------------------------------------------------------------------------- dataset
    def dataset_info(self) -> Dict[str, Any]:
        return dataset_info(self._metadata_dir)

    def preprocessing_config(self, dataset: Optional[str]) -> Optional[Dict[str, Any]]:
        return preprocessing_config(self._metadata_dir, dataset)

    # ----------------------------------------------------------------------------------------------- query
    def tokens(self, query: str, dataset: Optional[str]) -> List[str]:
        """Tokens for the lexical models, using the dataset's saved preprocessing config."""
        payload: Dict[str, Any] = {"text": query, "mode": "query"}
        config = self.preprocessing_config(dataset)
        if config:
            payload["config"] = config

        body = self._client.post(
            f"{self._settings.preprocessing_url}/preprocess",
            payload,
            timeout=self._settings.preprocessing_timeout,
            service="preprocessing",
        )
        tokens = [str(token) for token in (body.get("tokens") or [])]
        if not tokens:
            logger.info("Preprocessing left no tokens; sending the query text as it is")
        return tokens

    def refine(
        self,
        query: str,
        history: Optional[List[str]] = None,
        enabled: bool = True,
        session_id: str = "default",
        is_evaluation: bool = False,
        dataset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask the refinement service; it owns the response contract, we forward it."""
        body = self._client.post(
            f"{self._settings.query_refinement_url}/refine",
            {
                "query": query,
                "history": list(history or []),
                "enabled": enabled,
                "session_id": session_id,
                "is_evaluation": is_evaluation,
                "dataset": dataset,
            },
            timeout=self._settings.refinement_timeout,
            service="query_refinement",
        )
        logger.info("Refinement: %r -> %r", query[:60], str(body.get("refined_query", query))[:60])
        return body

    def _prepare(self, request: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """The query to retrieve with, and the refinement payload when one was applied."""
        query = str(getattr(request, "query", "") or "")
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not getattr(request, "enable_refinement", False):
            return query, None

        refinement = self.refine(
            query,
            getattr(request, "query_history", None),
            True,
            dataset=getattr(request, "dataset", None),
        )
        return str(refinement.get("refined_query") or query), refinement

    # ----------------------------------------------------------------------------------------------- search
    def _retrieve(self, path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        body = self._client.post(
            f"{self._settings.retrieval_url}{path}", payload, timeout=timeout, service="retrieval"
        )
        logger.info("%s returned %s results", path, body.get("total", len(body.get("results") or [])))
        return body

    @staticmethod
    def _with_refinement(results: Dict[str, Any], refinement: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if refinement:
            results["refinement"] = refinement
        return results

    def _payload_common(self, request: Any, query: str) -> Dict[str, Any]:
        dataset = getattr(request, "dataset", None)
        payload: Dict[str, Any] = {
            "query_text": query,
            "query_tokens": self.tokens(query, dataset),
            "include_text": True,
            "bm25_k1": getattr(request, "bm25_k1", 1.5),
            "bm25_b": getattr(request, "bm25_b", 0.75),
        }
        if dataset:
            payload["dataset"] = dataset
        return payload

    def search_basic(self, request: Any) -> Dict[str, Any]:
        """One model, one index."""
        query, refinement = self._prepare(request)
        payload = self._payload_common(request, query)
        payload.update(
            {
                "model": getattr(request, "model", "tfidf"),
                "top_k": getattr(request, "top_k", 50),
            }
        )
        return self._with_refinement(
            self._retrieve("/retrieve", payload, self._settings.retrieval_timeout), refinement
        )

    def search_hybrid_serial(self, request: Any) -> Dict[str, Any]:
        """Stage 1 retrieves candidates, stage 2 re-ranks them."""
        query, refinement = self._prepare(request)
        payload = self._payload_common(request, query)
        payload.update(
            {
                "first_model": getattr(request, "first_model"),
                "second_model": getattr(request, "second_model"),
                "first_stage_k": getattr(request, "first_stage_k", 1000),
                "final_top_k": getattr(request, "final_top_k", 50),
            }
        )
        return self._with_refinement(
            self._retrieve("/retrieve/hybrid/serial", payload, self._settings.hybrid_timeout), refinement
        )

    def search_hybrid_parallel(self, request: Any) -> Dict[str, Any]:
        """Every model retrieves on its own, then the lists are fused."""
        query, refinement = self._prepare(request)
        payload = self._payload_common(request, query)
        payload.update(
            {
                "models": getattr(request, "models"),
                "fusion_method": getattr(request, "fusion_method", "rrf"),
                "model_weights": getattr(request, "model_weights", None),
                "top_k": getattr(request, "top_k", 50),
                "candidates_per_model": getattr(request, "candidates_per_model", 1000),
                "rrf_k": getattr(request, "rrf_k", 60),
            }
        )
        return self._with_refinement(
            self._retrieve("/retrieve/hybrid/parallel", payload, self._settings.hybrid_timeout), refinement
        )
