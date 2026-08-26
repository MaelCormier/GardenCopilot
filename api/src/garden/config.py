"""
config.py — Toutes les constantes en un seul endroit.

Pour l'instant : os.environ brut.
En P1 (endpoint), on remplacera l'intérieur par pydantic-settings
SANS toucher aux autres modules (ils importent juste ces noms).
"""

import os

QDRANT_URL  = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION  = os.environ.get("QDRANT_COLLECTION", "jardinage")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
GROQ_MODEL  = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K       = int(os.environ.get("TOP_K", "3"))   # int() car os.environ rend une str