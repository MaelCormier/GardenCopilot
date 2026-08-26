# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from garden.rag import answer

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

app = FastAPI()

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    response, sources = answer(req.question)
    return AskResponse(answer=response, sources=sources)