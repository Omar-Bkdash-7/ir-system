from fastapi import FastAPI

app = FastAPI(title="Ranking Service")

@app.get("/health")
def health():
    return {"status": "ok", "service": "ranking"}
