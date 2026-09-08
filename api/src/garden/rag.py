"""
rag.py — Le chef d'orchestre : assemble R + A + G.

Contient l'Augmentation (build_prompt) et le pipeline complet (answer).
C'est CE module que main.py importera pour l'endpoint /ask.
"""

from garden.config import MAX_HISTORY, SYSTEM_PROMPT
from garden.embeddings import encode
from garden.llm import generate
from garden.vectorstore import search

# --- A : le garde-fou strict (ne répond QUE depuis le contexte) ---


def build_prompt(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> list[dict]:
    context = "\n\n".join(c["text"] for c in chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history[-MAX_HISTORY:])

    messages.append({"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {question}"})
    return messages


def answer(question: str) -> tuple[str, list[str]]:
    chunks = search(question)
    context = "\n\n".join(c["text"] for c in chunks)
    sources = list(dict.fromkeys(c["source"] for c in chunks))
    prompt = build_prompt(question, context)
    return generate(prompt), sources


if __name__ == "__main__":
    q = "Quand faut-il tailler un rosier ?"
    print(f"Q: {q}\n")
    print("R:", answer(q))
