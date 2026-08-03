from fastapi import FastAPI

app = FastAPI(title="Garden Copilot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
