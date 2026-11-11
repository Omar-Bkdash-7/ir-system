"""Datasets as evaluation needs them: queries, qrels, and the queries worth scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetSplit:
    """A dataset's queries and their relevance judgements, already in memory."""

    name: str
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]

    @property
    def evaluable_queries(self) -> List[str]:
        """Query ids that have at least one relevant document (others cannot be scored)."""
        return [query_id for query_id in self.queries if self.qrels.get(query_id)]

    def relevant_count(self, query_id: str) -> int:
        return sum(1 for grade in self.qrels.get(query_id, {}).values() if grade > 0)

    def limited(self, limit: Optional[int]) -> "DatasetSplit":
        """First ``limit`` evaluable queries (``None`` keeps every one)."""
        if not limit:
            return self
        keep = self.evaluable_queries[: int(limit)]
        return DatasetSplit(
            name=self.name,
            queries={query_id: self.queries[query_id] for query_id in keep},
            qrels={query_id: self.qrels[query_id] for query_id in keep},
        )
