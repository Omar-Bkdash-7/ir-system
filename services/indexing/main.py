from fastapi import FastAPI

app = FastAPI(title="Indexing Service")

@app.get("/health")
def health():
    return {"status": "ok", "service": "indexing"}
