"""Client for the Preprocessing microservice (text -> tokens preview)."""
from __future__ import annotations

import config
from api.client import APIResponse, health_check, post


def preprocess(text: str, preprocessing_config: dict) -> APIResponse:
    """
    POST /preprocess

    Body:   {"text": str, "config": PreprocessingConfig}
    Returns: {"tokens": [str, ...], "original": str}
    """
    return post(
        f"{config.PREPROCESSING_URL}/preprocess",
        json_body={"text": text, "config": preprocessing_config},
    )


def health() -> APIResponse:
    return health_check(config.PREPROCESSING_URL)