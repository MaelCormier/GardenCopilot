"""
rag_demo.py — Premier RAG bout-en-bout (R → A → G).

But : VOIR le RAG marcher en un script linéaire.
Chaque fonction est déjà rangée pour migrer sans effort vers son module :
    search()      -> vectorstore.py
    build_prompt()-> rag.py
    generate()    -> llm.py
    answer()      -> rag.py (orchestration)
"""

import os

from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --- Config (os.environ brut ici ; pydantic-settings viendra avec l'endpoint) ---
QDRANT_URL  = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION  = os.environ.get("QDRANT_COLLECTION", "jardinage")        # aligné sur l'indexation
EMBED_MODEL = os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")  # MÊME modèle qu'à l'index
GROQ_MODEL  = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")     # modèle production actif (l'ancien llama-3.3-70b est déprécié)
TOP_K       = 3

# --- Init : chargé une fois, pas à chaque requête ---
embedder = SentenceTransformer(EMBED_MODEL)   # -> embeddings.py
qdrant   = QdrantClient(url=QDRANT_URL)        # -> vectorstore.py
groq     = Groq()                              # lit GROQ_API_KEY depuis l'env -> llm.py


# ==================== R : Retrieval ====================
def search(question: str, top_k: int = TOP_K) -> list[str]:
    """question -> vecteur -> top_k chunks Qdrant.
    MiniLM : pas de préfixe, on encode brut (comme à l'indexation)."""
    vector = embedder.encode(question).tolist()
    hits = qdrant.query_points(          # query_points = remplaçant moderne de client.search()
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
    ).points
    return [h.payload["texte"] for h in hits]   # clé "texte", comme ton upsert


# ==================== A : Augmentation ====================
SYSTEM_PROMPT = (
    "Tu es un assistant jardinage. "
    "Réponds UNIQUEMENT à partir du CONTEXTE fourni. "
    "Si l'information n'y figure pas, réponds exactement : "
    '"Je ne trouve pas cette information dans mes documents." '
    "N'invente jamais, ne complète pas avec des connaissances externes."
)

def build_prompt(question: str, chunks: list[str]) -> list[dict]:
    """Fusionne consigne (system) + contexte Qdrant + question (user) en messages Groq."""
    context = "\n\n---\n\n".join(chunks)
    user_content = f"CONTEXTE:\n{context}\n\nQUESTION:\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ==================== G : Generation ====================
def generate(messages: list[dict]) -> str:
    """Appel Groq. temperature basse = factuel, colle au contexte."""
    resp = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ==================== Orchestration R → A → G ====================
def answer(question: str) -> str:
    chunks   = search(question)                # R
    messages = build_prompt(question, chunks)  # A
    return generate(messages)                  # G


if __name__ == "__main__":
    q = "Quand faut-il tailler un rosier ?"
    print(f"Q: {q}\n")

    # Debug utile en mode strict : voir CE QUI a été récupéré avant la génération
    chunks = search(q)
    print(f"[{len(chunks)} chunks récupérés]")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c[:80]}...")
    print()

    print("R:", answer(q))