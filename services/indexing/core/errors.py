"""Errors the HTTP layer of the indexing service turns into status codes.

`common.exception_handlers` already maps ValueError to 400 and FileNotFoundError to
404; the invalid id / index type errors subclass ValueError, so they need nothing
more. A failed build is different: it is a plain dict result, and the route must not
answer 200 with it.
"""
from __future__ import annotations

from typing import Optional

from services.indexing.errors import IndexingError

#: Why a build failed decides the status: a missing upload or a bad config is the caller's fault.
_STATUS_BY_REASON = {
    "dataset_not_in_database": 400,
    "invalid_preprocessing_config": 400,
}


class BuildFailedError(IndexingError):
    """A build answered `{"status": "error"}`; the route has to say so instead of returning 200."""

    def __init__(self, message: str, reason: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason = reason

    @property
    def status_code(self) -> int:
        return _STATUS_BY_REASON.get(self.reason or "", 500)
