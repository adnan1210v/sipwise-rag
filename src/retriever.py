"""
Schritt 5 der Pipeline: relevante Chunks ABRUFEN (Retrieval).

Das ist das "R" in RAG. Ablauf:
  1. Nutzerfrage in einen Vektor umwandeln (gleiches Modell wie die Dokumente!).
  2. In ChromaDB die top_k ähnlichsten Chunks suchen.
  3. Diese Chunks zurückgeben – sie dienen gleich als KONTEXT fürs LLM.
"""

from . import config
from .embeddings import embed_query
from .vector_store import get_collection, query


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """
    Findet die relevantesten Chunks zu einer Frage.

    top_k=None -> nimmt den Standardwert aus der config (TOP_K).
    """
    if top_k is None:
        top_k = config.TOP_K

    # Frage -> Vektor (Schritt 5a)
    question_vector = embed_query(question)

    # Vektor -> ähnlichste Chunks aus der DB (Schritt 5b)
    collection = get_collection()
    hits = query(collection, question_vector, top_k)
    return hits
