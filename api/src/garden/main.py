# src/garden/main.py
from fastapi import FastAPI
from pydantic import BaseModel

from garden.graph import answer

app = FastAPI(title="Garden Copilot")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = answer(req.question)
    return AskResponse(answer=result["answer"], sources=result["sources"])
