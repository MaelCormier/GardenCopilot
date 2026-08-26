"""
rag.py — Le chef d'orchestre : assemble R + A + G.

Contient l'Augmentation (build_prompt) et le pipeline complet (answer).
C'est CE module que main.py importera pour l'endpoint /ask.
"""

from garden.llm import generate
from garden.vectorstore import search

# --- A : le garde-fou strict (ne répond QUE depuis le contexte) ---
SYSTEM_PROMPT = (
    "Tu es un assistant jardinage. "
    "Réponds UNIQUEMENT à partir du CONTEXTE fourni. "
    "Si l'information n'y figure pas, réponds exactement : "
    '"Je ne trouve pas cette information dans mes documents." '
    "N'invente jamais, ne complète pas avec des connaissances externes."
)


def build_prompt(question: str, chunks: list[str]) -> list[dict]:
    """Fusionne consigne + chunks (contexte) + question en messages Groq."""
    context = "\n\n---\n\n".join(chunks)
    user_content = f"CONTEXTE:\n{context}\n\nQUESTION:\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def answer(question: str) -> str:
    """Le RAG complet en 3 temps."""
    chunks   = search(question)                # R
    messages = build_prompt(question, chunks)  # A
    return generate(messages)                  # G


if __name__ == "__main__":
    q = "Quand faut-il tailler un rosier ?"
    print(f"Q: {q}\n")
    print("R:", answer(q))