# src/garden/graph.py
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from garden.config import MAX_ATTEMPTS
from garden.llm import generate
from garden.rag import build_prompt
from garden.vectorstore import search


# ── 1. State ──────────────────────────────────────────────────
class RagState(TypedDict):
    question: str
    chunks: list[dict]
    answer: str
    sources: list[str]
    relevant: bool
    attempts: int


# ── 2. Nœuds ──────────────────────────────────────────────────
def retrieve(state: RagState) -> dict:
    return {"chunks": search(state["question"])}


def grade(state: RagState) -> dict:
    context = "\n\n".join(c["text"] for c in state["chunks"])
    prompt = [
        {
            "role": "system",
            "content": "Le contexte contient-il l'information nécessaire pour "
            "répondre à la question ? Une simple proximité de thème ne suffit "
            "pas : il faut la donnée précise demandée. Dans le doute, réponds "
            "'non'. Réponds UNIQUEMENT par 'oui' ou 'non'.",
        },
        {"role": "user", "content": f"Question: {state['question']}\n\nContexte:\n{context}"},
    ]
    verdict = generate(prompt).strip().lower()
    return {"relevant": verdict.startswith("oui")}


def rewrite(state: RagState) -> dict:
    prompt = [
        {
            "role": "system",
            "content": "Reformule la question pour améliorer une recherche "
            "documentaire. Utilise des synonymes et des termes plus explicites. "
            "Réponds UNIQUEMENT par la question reformulée.",
        },
        {"role": "user", "content": state["question"]},
    ]
    return {
        "question": generate(prompt).strip(),
        "attempts": state["attempts"] + 1,
    }


def generate_node(state: RagState) -> dict:
    sources = list(dict.fromkeys(c["source"] for c in state["chunks"]))
    prompt = build_prompt(state["question"], state["chunks"])
    return {"answer": generate(prompt), "sources": sources}


# ── 3. Décision ───────────────────────────────────────────────
def decide(state: RagState) -> str:
    if state["relevant"]:
        return "generate"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "generate"
    return "rewrite"


# ── 4. Assemblage ─────────────────────────────────────────────
def build_graph():
    builder = StateGraph(RagState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", decide)
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    return builder.compile()


graph = build_graph()


# ── 5. Test isolé ─────────────────────────────────────────────
def answer(question: str) -> dict:
    """Point d'entrée : question → {answer, sources}."""
    return graph.invoke({"question": question, "attempts": 0})


if __name__ == "__main__":
    result = answer("À quelle fréquence arroser mes tomates ?")
    print(result["answer"], "\n\nSources:", result["sources"])
