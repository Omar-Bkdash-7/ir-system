import httpx
from common.config import get_config
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

QUERY_REFINEMENT_SERVICE_URL = get_config("services.query_refinement_url", "http://localhost:8005")

def refine_query(
    query: str, 
    history: List[str], 
    enabled: bool = True,
    is_evaluation: bool = False
) -> Dict:
    """
    Refine query using the query refinement service.
    
    Args:
        query: Query text to refine
        history: List of previous queries for context
        enabled: Whether refinement is enabled
        is_evaluation: If True, only apply deterministic operations (spelling)
    """
    if not enabled:
        return {
            "original_query": query,
            "refined_query": query,
            "corrections": {},
            "expansions": {},
            "history_boosts": {},
            "confidence": 0.0,
            "changes_applied": [],
            "semantic_drift": 0.0,
            "applied": False
        }
    
    mode = "evaluation" if is_evaluation else "search"
    logger.info(f"Calling query refinement service (mode={mode}): query='{query[:50]}...'")
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{QUERY_REFINEMENT_SERVICE_URL}/refine",
                json={
                    "query": query,
                    "history": history,
                    "enabled": enabled,
                    "is_evaluation": is_evaluation
                }
            )
            response.raise_for_status()
            data = response.json()
            
            refined = data.get('refined_query', query)
            changes = data.get('changes_applied', [])
            logger.info(
                f"Query refinement completed: '{refined[:50]}...' "
                f"(changes: {changes}, confidence: {data.get('confidence', 0):.2f})"
            )
            return data
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during query refinement: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error during query refinement: {e}", exc_info=True)
        raise

