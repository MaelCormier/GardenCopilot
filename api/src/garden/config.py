# src/garden/config.py
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Constantes de code (fixes, pas d'env) ──────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # /app
DATA_DIR = BASE_DIR / "data"

COLLECTION = "jardinage"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
MAX_WORDS = 100
MIN_WORDS = 8
TOP_K = 5
MAX_ATTEMPTS = 2
MAX_HISTORY = 6  # 3 tours (user + assistant)

GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "Tu es un assistant jardinage. "
    "Réponds UNIQUEMENT à partir du CONTEXTE fourni. "
    "Si l'information n'y figure pas, réponds exactement : "
    '"Je ne trouve pas cette information dans mes documents." '
    "N'invente jamais, ne complète pas avec des connaissances externes."
)


# ── Config d'environnement (validée au boot) ───────────────────
class Settings(BaseSettings):
    groq_api_key: str
    qdrant_url: str = "http://qdrant:6333"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()  # instancié une fois, importé partout
