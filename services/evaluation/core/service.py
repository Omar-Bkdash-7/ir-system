"""Orchestration of one evaluation run.

Framework-free on purpose: the request is read with ``getattr``, results are plain
dicts, and every dependency (dataset loader, storage, HTTP client, query pipeline)
is injected — so the whole run can be exercised in a test with no FastAPI, no
PostgreSQL, no dataset download and no retrieval service.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from services.evaluation.config import EvaluationSettings, get_settings
from services.evaluation.engines.dataset_loader import DatasetLoader
from services.evaluation.schemas import (
    EvaluationMetrics,
    EvaluationResponse,
    QueryEvaluationResult,
)
from services.evaluation.services.results_storage import ResultsStorage

from .clients import JsonClient
from .names import is_hybrid, require_fusion, require_model
from .query import QueryPipeline
from .reports import latest_metrics_by_model, per_query_analysis, refinement_comparison
from .runner import RunResult, Retrieve, run_over_queries

logger = logging.getLogger(__name__)

#: Datasets offered by /datasets for the frontend dropdown. Evaluation itself is
#: not limited to this list: any ir_datasets name with qrels can be evaluated.
CATALOG_DATASETS = (
    "beir/quora/test",
    "beir/trec-covid",
    "beir/nfcorpus",
    "beir/scifact",
    "beir/fiqa",
)


class EvaluationService:
    """Run evaluations and answer the reporting endpoints."""

    def __init__(
        self,
        settings: EvaluationSettings,
        loader: DatasetLoader,
        storage: ResultsStorage,
        pipeline: QueryPipeline,
        client: Optional[JsonClient] = None,
    ) -> None:
        self._settings = settings
        self._loader = loader
        self._storage = storage
        self._pipeline = pipeline
        self._client = client or JsonClient(
            max_retries=settings.max_retries, backoff_seconds=settings.retry_backoff_seconds
        )

    @classmethod
    def create(cls, settings: Optional[EvaluationSettings] = None) -> "EvaluationService":
        settings = settings or get_settings()
        client = JsonClient(max_retries=settings.max_retries, backoff_seconds=settings.retry_backoff_seconds)
        pipeline = QueryPipeline(
            client,
            preprocessing_url=settings.preprocessing_url,
            refinement_url=settings.query_refinement_url,
            preprocessing_timeout=settings.preprocessing_timeout_seconds,
            refinement_timeout=settings.refinement_timeout_seconds,
        )
        return cls(settings, DatasetLoader(), ResultsStorage(), pipeline, client)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------------------------------------- run
    def evaluate(self, request: Any) -> EvaluationResponse:
        """Score one model (or one hybrid combination) on one dataset."""
        model = require_model(getattr(request, "model", None))
        dataset = self._require_dataset(getattr(request, "dataset_name", None))
        top_k = self._settings.top_k(getattr(request, "top_k", None))
        split = self._loader.load(dataset).limited(self._settings.query_limit(getattr(request, "query_limit", None)))

        refine = bool(getattr(request, "enable_refinement", False))
        logger.info(
            "Evaluating model=%s dataset=%s top_k=%d queries=%d refinement=%s workers=%d",
            model, dataset, top_k, len(split.evaluable_queries), refine, self._settings.workers,
        )

        run = run_over_queries(
            split,
            prepare=lambda text: self._pipeline.prepare(text, dataset=dataset, refine=refine),
            retrieve=self._retriever(model, dataset, top_k, request),
            top_k=top_k,
            label=f"{model}@{dataset}",
            workers=self._settings.workers,
        )

        metrics = run.metrics
        self._save(model, dataset, top_k, metrics, request, run)

        return EvaluationResponse(
            model=model,
            top_k=top_k,
            metrics=EvaluationMetrics(
                map=metrics["map"],
                ndcg=metrics["ndcg"],
                recall=metrics["recall"],
                precision=metrics["precision"],
                num_queries=metrics["num_queries"],
            ),
            query_results=self._per_query_results(run) if getattr(request, "include_per_query", False) else None,
            warnings=run.warnings() or None,
        )

    # ------------------------------------------------------------------------------------------------- reporting
    def get_available_datasets(self) -> List[str]:
        return list(CATALOG_DATASETS)

    def get_evaluation_history(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return self._storage.get_latest_results(model=model, dataset=dataset, limit=limit)

    def compare_models(self, dataset: str, top_k: int = 100) -> List[Dict[str, Any]]:
        return self._storage.compare_models(dataset=dataset, top_k=top_k)

    def get_latest_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Latest run of every model, keyed the way the frontend expects."""
        rows = self._storage.get_latest_results(limit=self._settings.history_limit)
        return latest_metrics_by_model(rows)

    def get_per_query_analysis(self) -> Dict[str, Any]:
        return per_query_analysis()

    def get_refinement_comparison(self) -> Dict[str, Any]:
        return refinement_comparison()

    def delete_result(self, result_id: int) -> bool:
        return self._storage.delete_result(result_id)

    def delete_results(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        before_date: Optional[str] = None,
    ) -> int:
        return self._storage.delete_by_filters(model=model, dataset=dataset, before_date=before_date)

    def clear_results(self) -> int:
        return self._storage.clear_all()

    # ------------------------------------------------------------------------------------------------- internals
    @staticmethod
    def _require_dataset(dataset_name: Optional[str]) -> str:
        name = (dataset_name or "").strip()
        if not name:
            raise ValueError("dataset_name is required")
        return name

    def _retriever(self, model: str, dataset: str, top_k: int, request: Any) -> Retrieve:
        """A callable that turns one prepared query into a ranked document list."""
        if model == "hybrid_serial":
            first, second = self._hybrid_models(request, expect=2)
            payload_base = {"first_model": first, "second_model": second}
            endpoint = f"{self._settings.retrieval_url}/retrieve/hybrid/serial"
        elif model == "hybrid_parallel":
            models = self._hybrid_models(request)
            payload_base = {"models": models, "fusion_method": require_fusion(getattr(request, "fusion_method", None))}
            endpoint = f"{self._settings.retrieval_url}/retrieve/hybrid/parallel"
        else:
            payload_base = {}
            endpoint = f"{self._settings.retrieval_url}/retrieve/{model}"

        k1 = getattr(request, "bm25_k1", None)
        b = getattr(request, "bm25_b", None)

        def retrieve(prepared) -> List[Dict[str, Any]]:
            payload: Dict[str, Any] = {
                "query_text": prepared.text,
                "top_k": top_k,
                "include_text": False,
                "dataset": dataset,
                **payload_base,
            }
            if model != "embedding":
                payload["query_tokens"] = list(prepared.tokens)
            if model == "bm25" or "bm25" in str(payload_base.get("models", "")) or "bm25" in (
                payload_base.get("first_model"),
                payload_base.get("second_model"),
            ):
                if k1 is not None:
                    payload["bm25_k1"] = k1
                if b is not None:
                    payload["bm25_b"] = b
            body = self._client.post(
                endpoint, payload, timeout=self._settings.retrieval_timeout_seconds, service="retrieval"
            )
            documents = body.get("results") or []
            return [document for document in documents if isinstance(document, dict) and "doc_id" in document]

        return retrieve

    @staticmethod
    def _hybrid_models(request: Any, *, expect: Optional[int] = None) -> List[str]:
        models = [require_model(name) for name in (getattr(request, "hybrid_models", None) or [])]
        if len(models) < 2:
            raise ValueError("Hybrid evaluation requires at least 2 models in hybrid_models")
        if expect and len(models) < expect:
            raise ValueError(f"Expected {expect} models for hybrid_models, got {len(models)}")
        for name in models:
            if is_hybrid(name):
                raise ValueError("hybrid_models must list single retrieval models (tfidf, bm25, embedding)")
        return models

    def _save(self, model: str, dataset: str, top_k: int, metrics: Dict[str, float],
              request: Any, run: RunResult) -> None:
        """Store the run; a storage problem is reported but never loses the result."""
        if not self._settings.save_results:
            logger.info("Result storage disabled by settings; not saving %s/%s", model, dataset)
            return

        metadata: Dict[str, Any] = {
            "dataset": dataset,
            "query_limit": getattr(request, "query_limit", None),
            "enable_refinement": bool(getattr(request, "enable_refinement", False)),
        }
        if is_hybrid(model):
            metadata["hybrid_models"] = getattr(request, "hybrid_models", None)
            metadata["fusion_method"] = getattr(request, "fusion_method", None)
        if model == "bm25":
            metadata["bm25_k1"] = getattr(request, "bm25_k1", None)
            metadata["bm25_b"] = getattr(request, "bm25_b", None)
        if run.failures:
            metadata["failed_queries"] = len(run.failures)

        try:
            self._storage.save_results(
                model=model,
                dataset=dataset,
                top_k=top_k,
                map_score=metrics["map"],
                ndcg_score=metrics["ndcg"],
                recall_score=metrics["recall"],
                precision_score=metrics["precision"],
                num_queries=metrics["num_queries"],
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - the metrics are still a valid answer
            logger.error("Could not store evaluation result for %s/%s: %s", model, dataset, exc, exc_info=True)

    @staticmethod
    def _per_query_results(run: RunResult) -> List[QueryEvaluationResult]:
        return [
            QueryEvaluationResult(
                query_id=query_id,
                query_text=record["query_text"],
                num_retrieved=record["num_retrieved"],
                num_relevant=record["num_relevant"],
                num_relevant_retrieved=record["num_relevant_retrieved"],
                ap=record["ap"],
                ndcg=record["ndcg"],
                recall=record["recall"],
                precision=record["precision"],
            )
            for query_id, record in run.results.items()
        ]
