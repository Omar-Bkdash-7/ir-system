from fastapi import FastAPI

app = FastAPI(title="Query Refinement Service")

@app.get("/health")
def health():
    return {"status": "ok", "service": "query_refinement"}
