"""Standard IR metrics, one formula each.

Graded relevance is supported throughout (a qrel grade of 2 counts twice as much
as 1 in nDCG, and as one relevant document in AP/recall/precision, which is what
trec_eval does).

    nDCG@k  = DCG@k / IDCG@k,  gain = 2^rel - 1,  discount = log2(rank + 1)
    AP      = (sum of precision@i over relevant hits at i) / (number of relevant)
    Recall  = relevant retrieved / relevant in qrels
    P@k     = relevant retrieved / retrieved
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence


def _gain(grade: int) -> float:
    return float(2 ** grade - 1)


def _discount(rank: int) -> float:
    """Rank is 1-based."""
    return math.log2(rank + 1)


def relevant_count(relevant_docs: Mapping[str, int]) -> int:
    return sum(1 for grade in relevant_docs.values() if grade > 0)


def retrieved_relevant(retrieved_doc_ids: Sequence[str], relevant_docs: Mapping[str, int]) -> int:
    return sum(1 for doc_id in retrieved_doc_ids if relevant_docs.get(doc_id, 0) > 0)


class MetricsCalculator:
    """Per-query IR metrics; all methods are pure and return 0.0 when undefined."""

    @staticmethod
    def calculate_ndcg(
        retrieved_doc_ids: List[str],
        relevant_docs: Dict[str, int],
        k: int,
    ) -> float:
        dcg = sum(
            _gain(relevant_docs.get(doc_id, 0)) / _discount(rank)
            for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1)
            if doc_id in relevant_docs
        )

        ideal_grades = sorted(
            (grade for grade in relevant_docs.values() if grade > 0), reverse=True
        )[:k]
        idcg = sum(_gain(grade) / _discount(rank) for rank, grade in enumerate(ideal_grades, start=1))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_average_precision(
        retrieved_doc_ids: List[str],
        relevant_docs: Dict[str, int],
    ) -> float:
        hits = 0
        precision_sum = 0.0

        for position, doc_id in enumerate(retrieved_doc_ids, start=1):
            if relevant_docs.get(doc_id, 0) > 0:
                hits += 1
                precision_sum += hits / position

        total_relevant = relevant_count(relevant_docs)
        return precision_sum / total_relevant if total_relevant > 0 else 0.0

    @staticmethod
    def calculate_recall(
        retrieved_doc_ids: List[str],
        relevant_docs: Dict[str, int],
    ) -> float:
        total_relevant = relevant_count(relevant_docs)
        if total_relevant <= 0:
            return 0.0
        return retrieved_relevant(retrieved_doc_ids, relevant_docs) / total_relevant

    @staticmethod
    def calculate_precision(
        retrieved_doc_ids: List[str],
        relevant_docs: Dict[str, int],
    ) -> float:
        if not retrieved_doc_ids:
            return 0.0
        return retrieved_relevant(retrieved_doc_ids, relevant_docs) / len(retrieved_doc_ids)

    @classmethod
    def calculate_all_metrics(
        cls,
        retrieved_doc_ids: List[str],
        relevant_docs: Dict[str, int],
        k: int,
    ) -> Dict[str, float]:
        return {
            "ndcg": cls.calculate_ndcg(retrieved_doc_ids, relevant_docs, k),
            "ap": cls.calculate_average_precision(retrieved_doc_ids, relevant_docs),
            "recall": cls.calculate_recall(retrieved_doc_ids, relevant_docs),
            "precision": cls.calculate_precision(retrieved_doc_ids, relevant_docs),
        }
