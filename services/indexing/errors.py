"""Exceptions raised by the indexing service.

They subclass the built-in exception the old code raised (ValueError), so every existing `except ValueError` keeps
working, and common.exception_handlers (ValueError -> HTTP 400) already maps them without any change.
"""


class IndexingError(Exception):
    """Base class of the indexing service's own exceptions."""


class InvalidDatasetIdError(IndexingError, ValueError):
    """The dataset id / name is empty, escapes its directory, or cannot be used as a PostgreSQL table name."""


class InvalidIndexTypeError(IndexingError, ValueError):
    """The index type is not one of tfidf / embedding / bm25."""
