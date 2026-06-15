from fastapi import FastAPI

app = FastAPI(title="Preprocessing Service")

@app.get("/health")
def health():
    return {"status": "ok", "service": "preprocessing"}
