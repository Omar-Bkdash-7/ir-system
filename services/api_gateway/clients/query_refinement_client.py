import httpx
from common.config import get_config
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

QUERY_REFINEMENT_SERVICE_URL = get_config("services.query_refinement_url", "http://localhost:8005")

def refine_query(query: str, history: List[str], enabled: bool = True) -> Dict:
    if not enabled:
        return {
            "original_query": query,
            "refined_query": query,
            "corrections": {},
            "synonyms_added": {},
            "history_terms_added": [],
            "history_similarity": 0.0,
            "applied": False
        }
    
    logger.info(f"Calling query refinement service: query='{query[:50]}...'")
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{QUERY_REFINEMENT_SERVICE_URL}/refine",
                json={
                    "query": query,
                    "history": history,
                    "enabled": enabled
                }
            )
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Query refinement completed: {data.get('refined_query', query)[:50]}...")
            return data
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during query refinement: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error during query refinement: {e}", exc_info=True)
        raise
