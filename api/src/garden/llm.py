"""
llm.py — Le G de RAG : générer une réponse à partir de messages.

Dépend seulement de config. Ne connaît ni Qdrant ni le modèle d'embedding.
C'est pour ça qu'on peut le tester sans lancer le retrieval.
"""

from collections.abc import AsyncIterator

from groq import AsyncGroq, Groq

from garden.config import GROQ_MODEL, settings

# Client créé une fois. Lit GROQ_API_KEY dans l'environnement.
_client = Groq(api_key=settings.groq_api_key)
_async_client = AsyncGroq(api_key=settings.groq_api_key)


def generate(messages: list[dict]) -> str:
    """messages (system + user) -> réponse rédigée. temperature basse = factuel."""
    resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content


async def generate_stream(messages: list[dict]) -> AsyncIterator[str]:
    stream = await _async_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


if __name__ == "__main__":
    import asyncio

    async def _demo():
        messages = [{"role": "user", "content": "Compte de 1 à 20."}]
        async for token in generate_stream(messages):
            print(token, end="", flush=True)
        print()

    asyncio.run(_demo())
