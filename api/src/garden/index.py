# src/garden/index.py
from qdrant_client.models import Distance, VectorParams, PointStruct

from garden import config
from garden.embeddings import encode
from garden.vectorstore import get_client

client = get_client() 


# ── 1. Lecture ────────────────────────────────────────────────
def read_docs() -> list[tuple[str, str]]:
    """[(nom_fichier, contenu), ...]"""
    return [(p.name, p.read_text(encoding="utf-8"))
            for p in sorted(config.DATA_DIR.glob("*.md"))]


# ── 2. Chunking ───────────────────────────────────────────────
def word_count(text: str) -> int:
    return len(text.split())


def split_on_headings(text: str) -> list[str]:
    """Découpe sur les titres Markdown, garde le titre avec sa section."""
    sections, current = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("#") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


def split_on_paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


def chunk_markdown(text: str) -> list[str]:
    chunks = []
    for section in split_on_headings(text):
        if word_count(section) <= config.MAX_WORDS:
            chunks.append(section)
        else:
            chunks.extend(split_on_paragraphs(section))
    return [c.strip() for c in chunks
            if c.strip() and word_count(c) >= config.MIN_WORDS]   # ← filtre


# ── 3. Collection (reset propre, pas de recreate_collection déprécié) ──
def reset_collection() -> None:
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config=VectorParams(
            size=config.EMBEDDING_DIM, distance=Distance.COSINE),
    )


# ── 4. Orchestration ──────────────────────────────────────────
def index() -> None:
    reset_collection()
    points, pid = [], 0
    for filename, text in read_docs():
        for chunk in chunk_markdown(text):
            points.append(PointStruct(
                id=pid,
                vector=encode(chunk),
                payload={"text": chunk, "source": filename},
            ))
            pid += 1
    client.upsert(collection_name=config.COLLECTION, points=points)
    print(f"{len(points)} chunks indexés depuis {config.DATA_DIR}")


if __name__ == "__main__":
    index()