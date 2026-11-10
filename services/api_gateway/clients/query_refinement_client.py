import logging
from typing import Dict, Optional, Sequence

import httpx

from common.config import get_config

logger = logging.getLogger(__name__)

QUERY_REFINEMENT_SERVICE_URL = get_config("services.query_refinement_url", "http://localhost:8005")


def refine_query(
    query: str,
    history: Sequence[str],
    enabled: bool = True,
    session_id: str = "default",
    is_evaluation: bool = False,
    dataset: Optional[str] = None,
) -> Dict:
    """Ask the query refinement service to refine ``query``.

    ``enabled=False`` is forwarded instead of answered here: the service owns the response contract, and a second
    copy of it in the gateway is exactly how the two drifted apart.
    """
    mode = "evaluation" if is_evaluation else "search"
    logger.info(f"Calling query refinement service (mode={mode}): query='{query[:50]}...'")

    with httpx.Client(timeout=10.0) as client:
        response = client.post(
            f"{QUERY_REFINEMENT_SERVICE_URL}/refine",
            json={
                "query": query,
                "history": list(history or []),
                "enabled": enabled,
                "session_id": session_id,
                "is_evaluation": is_evaluation,
                "dataset": dataset,
            },
        )
        response.raise_for_status()
        data = response.json()

    logger.info(
        f"Query refinement completed: '{str(data.get('refined_query', query))[:50]}...' "
        f"(changes: {data.get('changes_applied', [])}, confidence: {float(data.get('confidence') or 0):.2f})"
    )
    return data
