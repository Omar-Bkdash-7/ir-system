from fastapi import FastAPI
from services.ranking.ranking_service import rank_results
from services.ranking.schemas import RankingRequest, RankingResponse
from common.exception_handlers import register_exception_handlers
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Ranking Service")
register_exception_handlers(app)

@app.get("/health")
def health():
    return {"status": "ok", "service": "ranking"}

@app.post("/rank", response_model=RankingResponse)
def rank_endpoint(req: RankingRequest):
    ranked = rank_results(req)
    return RankingResponse(ranked_results=ranked, total=len(ranked))
