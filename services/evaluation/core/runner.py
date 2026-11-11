"""The evaluation loop, written once.

Single-model, hybrid-serial and hybrid-parallel runs differ only in *how* a query
is retrieved. Everything else — skipping queries without qrels, scoring a ranked
list, keeping per-query records, surviving one bad query, averaging the metrics —
happens here, so the three modes cannot drift apart.

Evaluation is dominated by one HTTP round trip per query, so queries are scored
concurrently (``workers``): the questions are independent and the retrieval
service answers several at a time. With ``workers=1`` the loop is the plain
sequential one. Results are always returned in dataset order, whatever order the
workers finished in.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from services.evaluation.engines.metrics_calculator import MetricsCalculator

from .dataset import DatasetSplit
from .query import PreparedQuery

logger = logging.getLogger(__name__)

#: A retriever: takes the prepared query, returns ranked documents (``doc_id`` keys).
Retrieve = Callable[[PreparedQuery], List[Dict[str, Any]]]
Prepare = Callable[[str], PreparedQuery]
#: What one query produced: its record, or why it could not be scored.
QueryOutcome = Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, str]]]


@dataclass
class RunResult:
    """Everything one evaluation run produced."""

    results: Dict[str, Dict[str, Any]]
    metrics: Dict[str, float]
    failures: List[Dict[str, str]] = field(default_factory=list)

    @property
    def num_queries(self) -> int:
        return len(self.results)

    def warnings(self) -> List[str]:
        """Short, human-readable summary of the queries that could not be scored."""
        if not self.failures:
            return []
        first = self.failures[0]
        return [
            f"{len(self.failures)} query/queries could not be evaluated "
            f"(first: {first['query_id']} — {first['error']})"
        ]


def per_query_record(
    query_text: str,
    documents: Sequence[Dict[str, Any]],
    relevant_docs: Dict[str, int],
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    """The stored shape of one scored query."""
    retrieved_ids = [document["doc_id"] for document in documents]
    return {
        "query_text": query_text,
        "retrieved": list(documents),
        "num_retrieved": len(retrieved_ids),
        "num_relevant": sum(1 for grade in relevant_docs.values() if grade > 0),
        "num_relevant_retrieved": sum(
            1 for doc_id in retrieved_ids if relevant_docs.get(doc_id, 0) > 0
        ),
        "ap": metrics["ap"],
        "ndcg": metrics["ndcg"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
    }


def aggregate_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of each per-query metric; zeros when nothing was scored."""
    if not records:
        return {"map": 0.0, "ndcg": 0.0, "recall": 0.0, "precision": 0.0, "num_queries": 0}

    count = len(records)
    return {
        "map": float(sum(record["ap"] for record in records) / count),
        "ndcg": float(sum(record["ndcg"] for record in records) / count),
        "recall": float(sum(record["recall"] for record in records) / count),
        "precision": float(sum(record["precision"] for record in records) / count),
        "num_queries": count,
    }


def _score_query(
    split: DatasetSplit,
    query_id: str,
    prepare: Prepare,
    retrieve: Retrieve,
    top_k: int,
    label: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    """Score one query; a failure is returned, never raised."""
    try:
        prepared = prepare(split.queries[query_id])
        documents = retrieve(prepared)
    except Exception as exc:  # noqa: BLE001 - one query must not abort the run
        logger.warning("%s: query %s failed: %s", label, query_id, exc)
        return None, {"query_id": query_id, "error": f"{type(exc).__name__}: {exc}"}

    if not documents:
        logger.info("%s: query %s returned no documents", label, query_id)
        return None, {"query_id": query_id, "error": "no documents retrieved"}

    relevant_docs = split.qrels.get(query_id, {})
    metrics = MetricsCalculator.calculate_all_metrics(
        [document["doc_id"] for document in documents], relevant_docs, top_k
    )
    return per_query_record(split.queries[query_id], documents, relevant_docs, metrics), None


def _outcomes_sequential(
    query_ids: List[str], split: DatasetSplit, prepare: Prepare, retrieve: Retrieve,
    top_k: int, label: str,
) -> Iterator[QueryOutcome]:
    for query_id in query_ids:
        record, failure = _score_query(split, query_id, prepare, retrieve, top_k, label)
        yield query_id, record, failure


def _outcomes_parallel(
    query_ids: List[str], split: DatasetSplit, prepare: Prepare, retrieve: Retrieve,
    top_k: int, label: str, workers: int,
) -> Iterator[QueryOutcome]:
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evaluation") as pool:
        futures = {
            pool.submit(_score_query, split, query_id, prepare, retrieve, top_k, label): query_id
            for query_id in query_ids
        }
        for future in as_completed(futures):
            query_id = futures[future]
            try:
                record, failure = future.result()
            except BaseException as exc:  # noqa: BLE001 - never let one worker kill the run
                record, failure = None, {"query_id": query_id, "error": f"{type(exc).__name__}: {exc}"}
            yield query_id, record, failure


def run_over_queries(
    split: DatasetSplit,
    *,
    prepare: Prepare,
    retrieve: Retrieve,
    top_k: int,
    label: str = "evaluation",
    workers: int = 1,
    progress_every: int = 50,
) -> RunResult:
    """
    Score every evaluable query of ``split``, ``workers`` queries at a time.

    A query that fails (retrieval error, empty result) is recorded as a failure and
    the run continues: one broken query must not hide the other 999.
    """
    started = time.perf_counter()
    query_ids = split.evaluable_queries
    skipped = len(split.queries) - len(query_ids)
    if skipped:
        logger.info("%s: skipping %d query/queries without qrels", label, skipped)

    workers = max(1, int(workers))
    parallel = workers > 1 and len(query_ids) > 1
    if parallel:
        logger.info("%s: evaluating %d queries with %d workers", label, len(query_ids), workers)

    outcomes = (
        _outcomes_parallel(query_ids, split, prepare, retrieve, top_k, label, workers)
        if parallel
        else _outcomes_sequential(query_ids, split, prepare, retrieve, top_k, label)
    )

    collected: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, str]] = []
    for position, (query_id, record, failure) in enumerate(outcomes, start=1):
        if record is not None:
            collected[query_id] = record
        if failure is not None:
            failures.append(failure)
        if progress_every and position % progress_every == 0:
            logger.info("%s: %d/%d queries processed", label, position, len(query_ids))

    # Dataset order, not completion order: two runs of the same input are identical.
    records = {query_id: collected[query_id] for query_id in query_ids if query_id in collected}
    metrics = aggregate_metrics(list(records.values()))
    took = time.perf_counter() - started
    logger.info(
        "%s finished: queries=%d scored=%d failed=%d workers=%d map=%.4f ndcg=%.4f took=%.1fs",
        label, len(query_ids), metrics["num_queries"], len(failures), workers if parallel else 1,
        metrics["map"], metrics["ndcg"], took,
    )
    return RunResult(results=records, metrics=metrics, failures=failures)
