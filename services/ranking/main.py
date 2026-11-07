"""Ranking service: the HTTP layer. It only validates requests and calls services.ranking.ranking_service."""
from fastapi import FastAPI

from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging
from services.ranking.ranking_service import fuse_result_lists, rank_results
from services.ranking.schemas import FusionRequest, FusionResponse, RankingRequest, RankingResponse

logger = setup_logging("ranking")

app = FastAPI(
    title="Ranking Service",
    description="Orders scored results (/rank) and fuses ranked lists from several models (/fuse). "
                "The retrieval service uses the same fusion code in-process (services.ranking.rankers).",
)
register_exception_handlers(app)


@app.get("/health")
def health():
    return {"status": "ok", "service": "ranking"}


@app.post("/rank", response_model=RankingResponse)
def rank_endpoint(req: RankingRequest):
    return rank_results(req.results, req.method, req.top_k)


@app.post("/fuse", response_model=FusionResponse)
def fuse_endpoint(req: FusionRequest):
    result_lists = [[{"doc_id": hit.doc_id, "score": hit.score} for hit in hits] for hits in req.result_lists]
    return fuse_result_lists(result_lists, req.method, req.weights, req.rrf_k, req.top_k)
