"""
vectorstore.py — Le R de RAG : retrouver les chunks les plus proches.

Dépend de embeddings (pour encoder la question) et de config.
Ne connaît pas Groq.
"""

from qdrant_client import QdrantClient

from garden.config import COLLECTION, QDRANT_URL, TOP_K
from garden.embeddings import encode

# Connexion ouverte une fois à l'import.
_client = QdrantClient(url=QDRANT_URL)


def search(question: str, top_k: int = TOP_K) -> list[str]:
    """question -> vecteur -> top_k chunks Qdrant (texte brut)."""
    vector = encode(question)
    hits = _client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
    ).points
    return [h.payload["texte"] for h in hits]


if __name__ == "__main__":
    chunks = search("taille des rosiers")
    print(f"[{len(chunks)} chunks]")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c[:80]}...")