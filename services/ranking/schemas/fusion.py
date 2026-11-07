from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from services.ranking.schemas.ranking_request import MAX_RESULTS, MAX_TOP_K

MAX_LISTS = 10
FUSION_METHODS = ["rrf", "combsum", "combmnz", "weighted"]


class Hit(BaseModel):
    # ids that were stored as numbers are handled as text, so a numeric id can never fail validation
    model_config = ConfigDict(coerce_numbers_to_str=True)

    doc_id: str = Field(description="Document id")
    score: float = Field(allow_inf_nan=False, description="Relevance score (its scale depends on the model that produced it)")


class FusionRequest(BaseModel):
    result_lists: List[List[Hit]] = Field(
        min_length=2,
        max_length=MAX_LISTS,
        description="Ranked result lists to fuse (one per model). Each list is ordered by score before it is fused",
    )
    method: str = Field(
        default="rrf",
        description="Fusion method: 'rrf', 'combsum', 'combmnz', 'weighted'",
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="One weight per list, only for method 'weighted' (scores are scaled by the weights; the order does not "
                    "change if every weight is multiplied by the same number)",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=100,
        description="RRF constant k",
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=MAX_TOP_K,
        description="Number of fused results to return",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v.lower() not in FUSION_METHODS:
            raise ValueError(f"Fusion method must be one of {FUSION_METHODS}, got '{v}'")
        return v.lower()

    @field_validator("result_lists")
    @classmethod
    def validate_list_sizes(cls, v: List[List[Hit]]) -> List[List[Hit]]:
        for position, hits in enumerate(v):
            if len(hits) > MAX_RESULTS:
                raise ValueError(f"result_lists[{position}] has {len(hits)} items; the maximum is {MAX_RESULTS}")
        return v

    @model_validator(mode="after")
    def validate_weights(self):
        if self.method == "weighted":
            if not self.weights:
                raise ValueError("weights is required when method='weighted'")
            if len(self.weights) != len(self.result_lists):
                raise ValueError(f"weights has {len(self.weights)} entries but there are {len(self.result_lists)} result lists")
            if any(w != w or w in (float("inf"), float("-inf")) or w < 0 for w in self.weights):
                raise ValueError("weights must be finite numbers >= 0")
            if sum(self.weights) <= 0:
                raise ValueError("at least one weight must be greater than 0")
        return self


class FusionResponse(BaseModel):
    fused_results: List[Hit] = Field(description="Fused documents, best first")
    total: int = Field(description="Number of fused results returned")
    method: str = Field(description="Fusion method that was used")
    took_ms: Optional[float] = Field(default=None, description="Server-side processing time in milliseconds")
