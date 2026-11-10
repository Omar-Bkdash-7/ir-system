"""Typed failures of the clustering service, mapped to HTTP status codes in main.create_app().

They all derive from IRSystemException so the shared handlers still answer for anything that slips through,
while each subclass keeps the status the caller needs to distinguish ("no clusters yet" is not "the service broke").
"""
from common.exceptions import IRSystemException


class ClusteringError(IRSystemException):
    """Base class of every clustering failure."""


class ClusteringUnavailableError(ClusteringError):
    """hdbscan / umap are not installed in this image (503)."""


class InvalidClusteringParameterError(ClusteringError):
    """A parameter is unknown or out of range: metric, min_cluster_size, n_keywords, dataset id (400)."""


class EmbeddingIndexMissingError(ClusteringError):
    """The embedding index of the dataset was never built (404)."""


class ClustersNotFoundError(ClusteringError):
    """The dataset has not been clustered yet (404)."""


class DocumentNotClusteredError(ClusteringError):
    """The dataset is clustered but this doc_id is not part of it (404)."""


class ClusteringInProgressError(ClusteringError):
    """A clustering run for this dataset is already writing its files (409)."""


class TextsUnavailableError(ClusteringError):
    """Document texts could not be read, so topics cannot be extracted. Clustering itself still succeeds."""
