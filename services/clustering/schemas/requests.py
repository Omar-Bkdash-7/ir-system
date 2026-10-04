from pydantic import BaseModel, Field
from typing import Optional


class ClusterRequest(BaseModel):
    dataset: str = Field(..., description="Dataset name to cluster")
    min_cluster_size: int = Field(
        default=5, description="Minimum cluster size for HDBSCAN"
    )
    min_samples: int = Field(default=3, description="Minimum samples for HDBSCAN")
    metric: str = Field(default="euclidean", description="Distance metric")
    force_recompute: bool = Field(
        default=False, description="Force recomputation of clusters"
    )


class TopicRequest(BaseModel):
    dataset: str = Field(..., description="Dataset name")
    n_keywords: int = Field(default=10, description="Number of keywords per topic")
    force_recompute: bool = Field(
        default=False, description="Force recomputation of topics"
    )
