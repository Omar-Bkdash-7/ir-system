from fastapi import FastAPI

app = FastAPI(title="Evaluation Service")

@app.get("/health")
def health():
    return {"status": "ok", "service": "evaluation"}
