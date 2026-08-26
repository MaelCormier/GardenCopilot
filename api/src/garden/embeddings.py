"""
embeddings.py — Le modèle qui transforme un texte en vecteur.

Responsabilité unique : encoder. Ne connaît ni Qdrant ni Groq.
"""

from sentence_transformers import SentenceTransformer

from garden.config import EMBED_MODEL

# Chargé UNE fois à l'import du module (≈470 Mo, coûteux : on ne le refait pas par requête).
_model = SentenceTransformer(EMBED_MODEL)


def encode(text: str) -> list[float]:
    """texte -> vecteur (liste de floats). .tolist() car Qdrant veut du Python natif, pas du numpy."""
    return _model.encode(text).tolist()


if __name__ == "__main__":
    vector = encode("Quand tailler un rosier ?")
    print(f"dimension du vecteur : {len(vector)}")   # doit afficher 384