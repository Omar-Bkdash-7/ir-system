"""Response bodies of the clustering service.

The fields the Streamlit page reads (n_clusters, n_noise, clusters[].size/topic, embeddings_2d, cluster_labels)
keep their names; everything else is additive: the resolved parameters, the document count and whether topics are
available at all (texts can be missing without the clustering itself failing).
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class TopicInfo(BaseModel):
    topic_id: int
    keywords: List[str] = Field(default_factory=list)
    representative_docs: Optional[List[str]] = None


class ClusterParams(BaseModel):
    """The parameters this run actually used (after defaults were applied)."""

    min_cluster_size: int
    min_samples: int
    metric: str
    n_keywords: int


class ClusterInfo(BaseModel):
    cluster_id: int
    size: int
    topic: Optional[TopicInfo] = None


class ClusterResponse(BaseModel):
    dataset: str
    n_clusters: int
    n_noise: int
    n_documents: int = 0
    clusters: List[ClusterInfo] = Field(default_factory=list)
    embeddings_2d: Optional[List[List[float]]] = Field(
        None, description="2D UMAP projection, one row per document (aligned with cluster_labels)"
    )
    cluster_labels: Optional[List[int]] = Field(
        None, description="Cluster label for each document (-1 = noise)"
    )
    params: Optional[ClusterParams] = None
    topics_available: bool = True
    topics_method: Optional[str] = Field(None, description="bertopic, tfidf or none")
    warnings: List[str] = Field(
        default_factory=list, description="Non-fatal problems, e.g. texts could not be read"
    )


class DocumentClusterInfo(BaseModel):
    doc_id: str
    cluster_id: int
    topic: Optional[TopicInfo] = None


class ClusterListResponse(BaseModel):
    datasets: List[str]


class ClusterStatusResponse(BaseModel):
    dataset: str
    status: str
    message: Optional[str] = None
    n_clusters: Optional[int] = None
    n_noise: Optional[int] = None
    n_documents: Optional[int] = None
    timestamp: Optional[str] = None
    params: Optional[ClusterParams] = None
    topics_available: Optional[bool] = None
    stale: bool = Field(
        default=False, description="The embedding index changed since this run, so the clusters are outdated"
    )
