"""
rag.py — Le chef d'orchestre : assemble R + A + G.

Contient l'Augmentation (build_prompt) et le pipeline complet (answer).
C'est CE module que main.py importera pour l'endpoint /ask.
"""

from garden.embeddings import encode
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


def build_prompt(question, chunks):
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    user_content = f"CONTEXTE:\n{context}\n\nQUESTION:\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


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
