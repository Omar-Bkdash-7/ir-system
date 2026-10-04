from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class TopicInfo(BaseModel):
    topic_id: int
    keywords: List[str]
    representative_docs: Optional[List[str]] = None


class ClusterInfo(BaseModel):
    cluster_id: int
    size: int
    topic: Optional[TopicInfo] = None


class ClusterResponse(BaseModel):
    dataset: str
    n_clusters: int
    n_noise: int
    clusters: List[ClusterInfo]
    embeddings_2d: Optional[List[List[float]]] = Field(
        None, description="2D UMAP projection for visualization"
    )
    cluster_labels: Optional[List[int]] = Field(
        None, description="Cluster label for each document"
    )


class DocumentClusterInfo(BaseModel):
    doc_id: str
    cluster_id: int
    topic: Optional[TopicInfo] = None


class ClusterListResponse(BaseModel):
    datasets: List[str]
