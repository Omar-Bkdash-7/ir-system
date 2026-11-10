"""Request bodies of the clustering service.

Every tuning knob is optional: None means "use the service default" (configs/config.yaml -> clustering:*), and the
response echoes the resolved values, so a caller knows what actually ran. Ranges are validated by pydantic, so an
out-of-range value is a 422/400 instead of an HDBSCAN traceback.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Keep in sync with config.HDBSCAN_METRICS.
ClusteringMetric = Literal["euclidean", "manhattan", "cosine"]


class ClusterRequest(BaseModel):
    dataset: str = Field(..., min_length=1, description="Dataset name to cluster (e.g. 'cranfield')")
    min_cluster_size: Optional[int] = Field(
        default=None, ge=2, description="Smallest group HDBSCAN accepts as a cluster (default: from config)"
    )
    min_samples: Optional[int] = Field(
        default=None, ge=1, description="Conservative-ness of HDBSCAN (default: from config)"
    )
    metric: Optional[ClusteringMetric] = Field(
        default=None, description="Distance metric; None uses the service default"
    )
    n_keywords: Optional[int] = Field(
        default=None, ge=1, le=50, description="Keywords per cluster topic (default: from config)"
    )
    include_visualization: bool = Field(
        default=True, description="Compute the 2D UMAP projection returned as embeddings_2d"
    )
    force_recompute: bool = Field(
        default=False, description="Recompute even if a run with these parameters and this index already exists"
    )

    @field_validator("dataset")
    @classmethod
    def _strip_dataset(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("dataset must not be blank")
        return cleaned
