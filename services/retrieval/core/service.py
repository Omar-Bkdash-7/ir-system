"""Orchestration of a retrieval request: retrieve, hybrid serial, hybrid parallel and status.

Framework-free on purpose (requests are read with getattr, results are plain dicts): the HTTP layer in main.py is a thin
adapter, and everything here can be tested without FastAPI, pydantic, PostgreSQL or a neural model.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.exceptions import InvalidQueryError, IRSystemException, RetrievalError

from ..config import RetrievalSettings, get_settings
from .compact import CompactCache
from .encoder import QueryEncoder
from .indexes import BaseIndex, Hits, SearchOptions
from .names import KINDS, LABELS
from .query import QueryContext, default_preprocess, preprocessor_for
from .store import IndexStore
from .texts import TextStore

logger = logging.getLogger(__name__)

FuseFunction = Callable[[List[List[Dict[str, Any]]], str, Optional[List[float]], int], List[Dict[str, Any]]]


def _default_fuse(results_list: List[List[Dict[str, Any]]], method: str, weights: Optional[List[float]], rrf_k: int) -> List[Dict[str, Any]]:
    from services.ranking.rankers import fuse_results  # in-process: the ranking package is a library for retrieval

    return fuse_results(results_list=results_list, method=method, weights=weights, rrf_k=rrf_k)


@dataclass
class SearchOutcome:
    results: List[Dict[str, Any]]
    model: str
    took_ms: float
    warnings: List[str] = field(default_factory=list)

    def to_response(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "results": self.results,
            "total": len(self.results),
            "model": self.model,
            "took_ms": round(self.took_ms, 2),
        }
        if self.warnings:
            body["warnings"] = self.warnings
        return body


class RetrievalService:
    def __init__(
        self,
        settings: RetrievalSettings,
        store: IndexStore,
        encoder: QueryEncoder,
        texts: TextStore,
        *,
        preprocess: Callable[[str], Sequence[str]] = default_preprocess,
        fuse: Optional[FuseFunction] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._encoder = encoder
        self._texts = texts
        self._preprocess = preprocess
        self._fuse = fuse or _default_fuse
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=settings.parallel_workers, thread_name_prefix="retrieval")

    @classmethod
    def create(cls, settings: Optional[RetrievalSettings] = None) -> "RetrievalService":
        settings = settings or get_settings()
        store = IndexStore(
            cache=CompactCache(settings.cache_dir),
            stale_check_seconds=settings.stale_check_seconds,
            memory_budget_mb=settings.index_memory_budget_mb,
            embedding_mmap=settings.embedding_mmap,
        )
        return cls(settings, store, QueryEncoder(settings.query_cache_size), TextStore(settings.db_pool_max))

    # ---------------------------------------------------------------------------------------------------------------
    # request helpers
    # ---------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _require_dataset(dataset: Optional[str]) -> str:
        if not dataset or not str(dataset).strip():
            raise InvalidQueryError("dataset parameter is required")
        return str(dataset)

    @staticmethod
    def _require_model(model: str) -> str:
        if model not in KINDS:
            raise InvalidQueryError(f"Unknown retrieval model: {model}")
        return model

    def _context(self, req: Any) -> QueryContext:
        return QueryContext(
            text=getattr(req, "query_text", None),
            tokens=getattr(req, "query_tokens", None),
            encoder=self._encoder,
            embedding_query_mode=self._settings.embedding_query_mode,
            preprocess=self._preprocess_for(getattr(req, "dataset", None)),
        )

    def _preprocess_for(self, dataset: Optional[str]) -> Callable[[str], Sequence[str]]:
        """An explicitly injected `preprocess` wins (tests); otherwise the config the dataset was indexed with."""
        if self._preprocess is not default_preprocess:
            return self._preprocess
        return preprocessor_for(dataset)

    def _options(self, req: Any) -> SearchOptions:
        k1 = getattr(req, "bm25_k1", None)
        b = getattr(req, "bm25_b", None)
        return SearchOptions(
            self._settings.bm25_default_k1 if k1 is None else float(k1),
            self._settings.bm25_default_b if b is None else float(b),
        )

    @staticmethod
    def _hits_to_dicts(ids: List[Any], scores: Sequence[float]) -> List[Dict[str, Any]]:
        return [{"doc_id": doc_id, "score": float(score)} for doc_id, score in zip(ids, scores)]

    def _results(self, index: BaseIndex, hits: Hits) -> List[Dict[str, Any]]:
        return self._hits_to_dicts(index.ids_for(hits.rows), hits.scores.tolist())

    def _attach_texts(self, results: List[Dict[str, Any]], dataset: str) -> List[str]:
        """Add 'text' to every result. A database problem degrades to empty texts (as before) but is reported."""
        if not results:
            return []
        try:
            texts = self._texts.fetch(dataset, [item["doc_id"] for item in results])
        except Exception as exc:
            logger.error("Could not fetch document texts for dataset %s: %s", dataset, exc, exc_info=True)
            for item in results:
                item["text"] = ""
            return [f"document texts unavailable ({type(exc).__name__})"]
        missing = 0
        for item in results:
            text = texts.get(item["doc_id"])
            if text is None:
                missing += 1
                text = ""
            item["text"] = text
        if missing:
            logger.warning("%d of %d documents have no text in the database (dataset %s)", missing, len(results), dataset)
        return []

    def _outcome(self, started: float, label: str, dataset: str, results: List[Dict[str, Any]], warnings: List[str]) -> SearchOutcome:
        took_ms = (time.perf_counter() - started) * 1000.0
        logger.info("retrieve model=%s dataset=%s hits=%d took_ms=%.1f", label, dataset, len(results), took_ms)
        return SearchOutcome(results, label, took_ms, warnings)

    # ---------------------------------------------------------------------------------------------------------------
    # single model
    # ---------------------------------------------------------------------------------------------------------------
    def retrieve(self, req: Any, model: Optional[str] = None) -> SearchOutcome:
        started = time.perf_counter()
        model = self._require_model(model or req.model)
        dataset = self._require_dataset(req.dataset)
        query = self._context(req)
        query.require(model)  # bad query -> 400 before any index is touched
        index = self._store.get(dataset, model)
        hits = index.search(index.prepare(query), req.top_k, self._options(req))
        results = self._results(index, hits)
        warnings = self._attach_texts(results, dataset) if req.include_text else []
        return self._outcome(started, model, dataset, results, warnings)

    # ---------------------------------------------------------------------------------------------------------------
    # hybrid serial: stage 1 finds candidates, stage 2 re-scores exactly those candidates with another model
    # ---------------------------------------------------------------------------------------------------------------
    def hybrid_serial(self, req: Any) -> SearchOutcome:
        started = time.perf_counter()
        dataset = self._require_dataset(req.dataset)
        first = self._require_model(req.first_model)
        second = self._require_model(req.second_model)
        label = f"hybrid_serial_{first}_{second}"
        if first == second:
            logger.warning("Hybrid serial uses the same model for both stages (%s): re-ranking cannot change the order", first)
        query = self._context(req)
        query.require(first)
        query.require(second)
        opts = self._options(req)

        first_index = self._store.get(dataset, first)
        first_plan = first_index.prepare(query)
        candidates = first_index.search(first_plan, req.first_stage_k, opts)
        if candidates.rows.size == 0:
            return self._outcome(started, label, dataset, [], [])

        second_index = first_index if second == first else self._store.get(dataset, second)
        if second_index is first_index:
            rows, second_plan = candidates.rows, first_plan
        else:
            rows = second_index.rows_for(first_index.ids_for(candidates.rows))
            second_plan = second_index.prepare(query)
        scores = second_index.score(second_plan, rows, opts)

        order = np.argsort(-scores, kind="stable")[: req.final_top_k]  # stable: equal scores keep the stage-1 order
        candidate_ids = first_index.ids_for(candidates.rows)
        results = self._hits_to_dicts([candidate_ids[i] for i in order.tolist()], scores[order].tolist())
        warnings = self._attach_texts(results, dataset) if req.include_text else []
        return self._outcome(started, label, dataset, results, warnings)

    # ---------------------------------------------------------------------------------------------------------------
    # hybrid parallel: every model retrieves on its own, then the result lists are fused
    # ---------------------------------------------------------------------------------------------------------------
    def _search_model(self, dataset: str, model: str, query: QueryContext, k: int, opts: SearchOptions) -> Tuple[BaseIndex, Hits]:
        index = self._store.get(dataset, model)
        return index, index.search(index.prepare(query), k, opts)

    def hybrid_parallel(self, req: Any) -> SearchOutcome:
        started = time.perf_counter()
        dataset = self._require_dataset(req.dataset)
        models = [self._require_model(model) for model in req.models]
        if len(models) < 2:
            raise InvalidQueryError("Parallel mode requires at least 2 models")
        label = f"hybrid_parallel_{'_'.join(models)}_{req.fusion_method}"
        query = self._context(req)
        for model in models:
            query.require(model)
        opts = self._options(req)

        futures = [self._executor.submit(self._search_model, dataset, m, query, req.candidates_per_model, opts) for m in models]
        results_list: List[List[Dict[str, Any]]] = []
        warnings: List[str] = []
        failures: List[BaseException] = []
        for model, future in zip(models, futures):
            try:
                index, hits = future.result()
                results_list.append(self._results(index, hits))
            except Exception as exc:
                known = isinstance(exc, (IRSystemException, FileNotFoundError, ValueError))  # expected: no traceback needed
                logger.log(logging.WARNING if known else logging.ERROR, "Model %s failed during hybrid parallel retrieval: %s",
                           model, exc, exc_info=not known)
                failures.append(exc)
                results_list.append([])
                warnings.append(f"{model} failed ({type(exc).__name__}): {exc}")
        if len(failures) == len(models):
            raise failures[0]  # nothing to fuse: surface the real error (and its HTTP status)
        if failures:
            warnings.append("results were fused from the remaining models only")

        weights = None
        if req.fusion_method == "weighted" and getattr(req, "model_weights", None):
            weights = [req.model_weights.get(model, 0.0) for model in models]
        try:
            fused = self._fuse(results_list, req.fusion_method, weights, req.rrf_k)
        except Exception as exc:
            logger.error("Fusion failed: %s", exc, exc_info=True)
            raise RetrievalError(f"Hybrid parallel retrieval failed: {exc}") from exc
        results = fused[: req.top_k]
        warnings += self._attach_texts(results, dataset) if req.include_text else []
        return self._outcome(started, label, dataset, results, warnings)

    # ---------------------------------------------------------------------------------------------------------------
    # status
    # ---------------------------------------------------------------------------------------------------------------
    def overall_status(self, dataset: Optional[str] = None) -> Dict[str, Any]:
        """Which models have an index. Nothing is loaded; the frontend's System Monitor reads `models`."""
        names = [dataset] if dataset else self._store.known_datasets()
        detail = {name: {kind: "ready" if self._store.is_built(name, kind) else "not_built" for kind in KINDS} for name in names}
        models = {kind: "ready" if any(state[kind] == "ready" for state in detail.values()) else "not_built" for kind in KINDS}
        return {
            "service": "retrieval",
            "message": "Retrievers loaded on-demand per dataset. /status returns every model for every known dataset.",
            "models": models,
            "datasets": detail,
            "loaded_indexes": self._store.loaded(),
        }

    # ---------------------------------------------------------------------------------------------------------------
    # warm-up and shutdown
    # ---------------------------------------------------------------------------------------------------------------
    def warm(self, spec: Sequence[str]) -> None:
        """Load indexes ahead of the first request. `spec` items: "dataset" (all built models) or "dataset:bm25|tfidf"."""
        for item in spec:
            dataset, _, kinds = item.partition(":")
            wanted = [kind.strip() for kind in kinds.split("|") if kind.strip()] or list(KINDS)
            for kind in wanted:
                try:
                    if kind not in KINDS or not self._store.is_built(dataset, kind):
                        continue
                    started = time.perf_counter()
                    self._store.get(dataset, kind).warm()
                    if kind == "embedding":
                        self._encoder.warm()
                    logger.info("Warmed %s index of %s in %.1fs", LABELS[kind], dataset, time.perf_counter() - started)
                except Exception:
                    logger.exception("Warm-up of %s/%s failed (the index will load on first use)", dataset, kind)

    def start_warmup(self) -> Optional[threading.Thread]:
        if not self._settings.warmup:
            return None
        thread = threading.Thread(target=self.warm, args=(self._settings.warmup,), name="retrieval-warmup", daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._texts.close()
