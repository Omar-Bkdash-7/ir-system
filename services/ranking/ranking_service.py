from services.ranking.rankers import rank_by_cosine
from services.ranking.schemas import RankingRequest
from common.config import get_config
import logging

logger = logging.getLogger(__name__)

def rank_results(request: RankingRequest) -> list[dict]:
    top_k = request.top_k
    if top_k is None:
        top_k = get_config("retrieval.top_k", 50)
    
    if request.method == "cosine":
        return rank_by_cosine(request.results, top_k)
    else:
        raise ValueError(f"Unknown ranking method: {request.method}")
