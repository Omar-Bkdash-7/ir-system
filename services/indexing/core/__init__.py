"""Core of the indexing service: the seams the HTTP layer talks to.

The heavy lifting (ir_datasets, PostgreSQL, scikit-learn, the indexers) stays in
`services/indexing/dataset/`, `indexers/` and `services/`, and is imported on demand
so importing this package stays cheap.
"""
from services.indexing.core.build import BuildService
from services.indexing.core.datasets import DatasetOperations
from services.indexing.core.errors import BuildFailedError
from services.indexing.core.indices import IndexAdminService
from services.indexing.core.service import IndexingService
from services.indexing.core.stats import StatsService

__all__ = [
    "BuildFailedError",
    "BuildService",
    "DatasetOperations",
    "IndexAdminService",
    "IndexingService",
    "StatsService",
]
