"""Deleting built indices (one type, or everything a dataset owns)."""
from __future__ import annotations

from typing import Any, Dict


class IndexAdminService:
    """Removes index files and drops the loaded copies from the cache."""

    def delete(self, index_type: str, dataset_name: str) -> Dict[str, Any]:
        from services.indexing.services.index_admin import delete_index

        return delete_index(index_type, dataset_name)

    def delete_all(self, dataset_name: str) -> Dict[str, Any]:
        from services.indexing.services.index_admin import delete_all_indices

        return delete_all_indices(dataset_name)
