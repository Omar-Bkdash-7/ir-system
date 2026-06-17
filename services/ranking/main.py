from fastapi import FastAPI
from services.ranking.ranking_service import rank_results
from services.ranking.schemas import RankingRequest, RankingResponse
from common.exception_handlers import register_exception_handlers
from common.logging_config import setup_logging

logger = setup_logging("ranking")

app = FastAPI(title="Ranking Service")
register_exception_handlers(app)

@app.get("/health")
def health():
    return {"status": "ok", "service": "ranking"}

@app.post("/rank", response_model=RankingResponse)
def rank_endpoint(req: RankingRequest):
    ranked = rank_results(req)
    return RankingResponse(ranked_results=ranked, total=len(ranked))
