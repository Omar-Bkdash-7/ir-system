"""Building indices: resolve which builder runs, and turn a failure result into an error.

The builders themselves (preprocessing pass, TF-IDF/BM25/embedding) live in
`services/indexing/services/build_service.py` — this module is the seam the HTTP
layer and the tests talk to, so a route never has to interpret a result dict.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from services.indexing.core.errors import BuildFailedError

logger = logging.getLogger(__name__)

#: kind -> name of the implementation in services.indexing.services.build_service
_IMPLEMENTATIONS = {
    "tfidf": "build_tfidf_index",
    "embedding": "build_embedding_index",
    "bm25": "build_bm25_index",
    "all": "build_all_indices",
}


class BuildService:
    """Runs one build (or all of them) and reports failure by raising."""

    def __init__(self, overrides: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None) -> None:
        #: kind -> callable; lets a test replace one builder without touching the filesystem
        self._overrides = dict(overrides or {})

    def _implementation(self, kind: str) -> Callable[..., Dict[str, Any]]:
        if kind in self._overrides:
            return self._overrides[kind]
        from services.indexing.services import build_service as implementation

        return getattr(implementation, _IMPLEMENTATIONS[kind])

    def tfidf(self, dataset_name: str, force_rebuild: bool = False,
              preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
        return self._run("tfidf", dataset_name, force_rebuild, preprocessing_config)

    def embedding(self, dataset_name: str, force_rebuild: bool = False,
                  preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
        return self._run("embedding", dataset_name, force_rebuild, preprocessing_config)

    def bm25(self, dataset_name: str, force_rebuild: bool = False,
             preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
        return self._run("bm25", dataset_name, force_rebuild, preprocessing_config)

    def all(self, dataset_name: str, force_rebuild: bool = False,
            preprocessing_config: Optional[dict] = None) -> Dict[str, Any]:
        return self._run("all", dataset_name, force_rebuild, preprocessing_config)

    def _run(self, kind: str, dataset_name: str, force_rebuild: bool,
             preprocessing_config: Optional[dict]) -> Dict[str, Any]:
        if not dataset_name or not dataset_name.strip():
            raise ValueError("dataset_name is required")
        result = self._implementation(kind)(dataset_name, force_rebuild, preprocessing_config)
        return self._finished(kind, dataset_name, result)

    @staticmethod
    def _finished(kind: str, dataset_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """A skipped or finished build is a 200; a failure is an error with the reason kept."""
        if isinstance(result, dict) and result.get("status") == "error":
            message = result.get("message") or f"{kind} index build failed"
            logger.error("Build failed for '%s' (%s): %s", dataset_name, result.get("reason"), message)
            raise BuildFailedError(message, reason=result.get("reason"))
        return result
