"""
llm.py — Le G de RAG : générer une réponse à partir de messages.

Dépend seulement de config. Ne connaît ni Qdrant ni le modèle d'embedding.
C'est pour ça qu'on peut le tester sans lancer le retrieval.
"""

from groq import Groq

from garden.config import settings, GROQ_MODEL

# Client créé une fois. Lit GROQ_API_KEY dans l'environnement.
_client = Groq(api_key=settings.groq_api_key)


def generate(messages: list[dict]) -> str:
    """messages (system + user) -> réponse rédigée. temperature basse = factuel."""
    resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    # Test isolé : pas besoin de Qdrant, juste un message bidon.
    print(generate([{"role": "user", "content": "Réponds 'ok' et rien d'autre."}]))