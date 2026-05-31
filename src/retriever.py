"""
Schritt 5 der Pipeline: relevante Chunks ABRUFEN (Retrieval).

Das ist das "R" in RAG. Ablauf:
  1. Nutzerfrage in einen Vektor umwandeln (gleiches Modell wie die Dokumente!).
  2. In ChromaDB die top_k ähnlichsten Chunks suchen.
  3. Diese Chunks zurückgeben – sie dienen gleich als KONTEXT fürs LLM.
"""

from . import config
from .embeddings import embed_query
from .query_expander import expand_query
from .vector_store import get_collection, query


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """
    Findet die relevantesten Chunks zu einer Frage.

    top_k=None -> nimmt den Standardwert aus der config (TOP_K).
    """
    if top_k is None:
        top_k = config.TOP_K

    collection = get_collection()

    # Statt nur die rohe Frage zu suchen, erzeugen wir optional mehrere
    # Suchvarianten. Das macht kurze deutsche Fragen und Tippfehler robuster,
    # während englische Fragen unverändert gut funktionieren.
    if config.QUERY_EXPANSION_ENABLED:
        variants = expand_query(
            question,
            max_variants=config.QUERY_EXPANSION_MAX_VARIANTS,
        )
    else:
        variants = [question]

    merged_hits: dict[tuple[str, int], dict] = {}
    fusion_scores: dict[tuple[str, int], float] = {}

    for variant_index, variant in enumerate(variants):
        # Frage/Suchvariante -> Vektor (Schritt 5a)
        question_vector = embed_query(variant)

        # Vektor -> ähnlichste Chunks aus der DB (Schritt 5b)
        variant_weight = 1.0 + min(variant_index, 3) * 0.05
        for rank, hit in enumerate(query(collection, question_vector, top_k), start=1):
            key = (hit["source"], hit["chunk_index"])
            # Reciprocal Rank Fusion: Ränge aus mehreren Suchvarianten sind
            # robuster vergleichbar als rohe Distanzen verschiedener Fragen.
            fusion_scores[key] = fusion_scores.get(key, 0.0) + variant_weight / (60 + rank)
            existing = merged_hits.get(key)
            if existing is None or hit["distance"] < existing["distance"]:
                hit["query_variant"] = variant
                merged_hits[key] = hit

    # Die Treffer aus allen Varianten werden per Rank-Fusion sortiert. Das ist
    # stabiler als rohe Distanzen zu vergleichen, weil jede Suchvariante eine
    # leicht andere semantische Frage ist.
    return sorted(
        merged_hits.values(),
        key=lambda hit: (-fusion_scores[(hit["source"], hit["chunk_index"])], hit["distance"]),
    )[:top_k]
