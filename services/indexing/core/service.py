"""The indexing service: the one object the HTTP layer depends on.

Each collaborator can be replaced, so a route can be exercised without a database,
an index on disk or ir_datasets.
"""
from __future__ import annotations

from typing import Optional

from services.indexing.config import IndexingSettings, get_settings
from services.indexing.core.build import BuildService
from services.indexing.core.datasets import DatasetOperations
from services.indexing.core.indices import IndexAdminService
from services.indexing.core.stats import StatsService


class IndexingService:
    """Dataset management, index builds, statistics and deletion behind one name."""

    def __init__(
        self,
        settings: Optional[IndexingSettings] = None,
        datasets: Optional[DatasetOperations] = None,
        builds: Optional[BuildService] = None,
        stats: Optional[StatsService] = None,
        indices: Optional[IndexAdminService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.datasets = datasets or DatasetOperations()
        self.builds = builds or BuildService()
        self.stats = stats or StatsService()
        self.indices = indices or IndexAdminService()

    @classmethod
    def create(cls, settings: Optional[IndexingSettings] = None) -> "IndexingService":
        return cls(settings=settings or get_settings())
