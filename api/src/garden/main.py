# src/garden/main.py
import json
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from garden.graph import answer, prepare
from garden.llm import generate_stream
from garden.rag import build_prompt

app = FastAPI(title="Garden Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[Message] = []


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = answer(req.question, [m.model_dump() for m in req.history])
    return AskResponse(answer=result["answer"], sources=result["sources"])


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    async def event_generator():
        prepared = prepare(req.question)
        chunks = prepared["chunks"]

        sources = sorted({c["source"] for c in chunks})
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        messages = build_prompt(
            prepared["question"],
            chunks,
            [m.model_dump() for m in req.history],
        )
        async for token in generate_stream(messages):
            yield f"data: {json.dumps({'token': token})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
