"""
Cross-service health aggregation.

Kept separate from the individual `api/*.py` clients because it fans out
across all seven services at once and is the one place that needs an
`st.cache_data` wrapper shared by both the always-visible sidebar and the
System Monitor page.
"""
from __future__ import annotations

import time
from typing import Callable, Dict

import streamlit as st

import config
import api.evaluation as evaluation
import api.gateway as gateway
import api.indexing as indexing
import api.preprocessing as preprocessing
import api.ranking as ranking
import api.retrieval as retrieval
from api.client import APIResponse, health_check

SERVICE_CHECKS: Dict[str, Callable[[], APIResponse]] = {
    "API Gateway": gateway.health,
    "Preprocessing": preprocessing.health,
    "Indexing": indexing.health,
    "Retrieval": retrieval.health,
    "Ranking": ranking.health,
    "Query Refinement": lambda: health_check(config.QUERY_REFINEMENT_URL),
    "Evaluation": evaluation.health,
}


def _to_dict(resp: APIResponse) -> dict:
    return {
        "ok": resp.ok,
        "elapsed_ms": resp.elapsed_ms,
        "error": resp.error,
        "status_code": resp.status_code,
        "checked_at": time.strftime("%H:%M:%S"),
    }


@st.cache_data(ttl=config.CACHE_TTL_HEALTH, show_spinner=False)
def check_all_services() -> Dict[str, dict]:
    """Cached fan-out health probe across every registered service."""
    return {name: _to_dict(check_fn()) for name, check_fn in SERVICE_CHECKS.items()}


def check_one_service(name: str) -> dict:
    """Uncached single-service probe, used by the System Monitor 'Test' button."""
    check_fn = SERVICE_CHECKS.get(name)
    if check_fn is None:
        return {"ok": False, "elapsed_ms": 0.0, "error": f"Unknown service: {name}",
                "status_code": None, "checked_at": time.strftime("%H:%M:%S")}
    return _to_dict(check_fn())


def clear_cache() -> None:
    check_all_services.clear()