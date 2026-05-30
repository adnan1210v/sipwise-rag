"""
Schritt 4 der Pipeline: Chunks + Vektoren SPEICHERN (und später suchen).

Dieser Modul kapselt ChromaDB. ChromaDB ist eine "Vektor-Datenbank": Sie
speichert nicht nur Text, sondern auch dessen Embedding-Vektor, und kann
blitzschnell die ähnlichsten Vektoren zu einer Frage finden ("Nearest
Neighbor Search").

WARUM ChromaDB?
- Läuft komplett lokal/offline, kein Server nötig (PersistentClient = Dateien
  auf der Festplatte). Perfekt für ein datenschutzbewusstes Offline-Projekt.
- Sehr einfache API, ideal zum Lernen.
- Speichert Metadaten (Quelle, chunk_index) gleich mit – damit bleibt die
  Nachvollziehbarkeit erhalten.
"""

import chromadb
from .config import CHROMA_DIR, COLLECTION_NAME


def get_collection():
    """
    Öffnet (oder erstellt) die ChromaDB-Collection auf der Festplatte.

    Eine "Collection" ist wie eine Tabelle. Wir nutzen die Cosine-Distanz als
    Ähnlichkeitsmaß – Standard für Text-Embeddings, weil es die RICHTUNG der
    Vektoren vergleicht (Bedeutung), nicht ihre Länge.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def reset_collection():
    """
    Löscht die Collection und legt sie neu an.

    Nützlich beim erneuten Einspeisen: So vermeiden wir Duplikate, wenn das
    Ingest-Skript mehrfach läuft.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # Collection existierte noch nicht – kein Problem.
        pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(collection, chunks: list[dict], embeddings: list[list[float]]):
    """
    Schreibt Chunks samt ihrer Embeddings in die Collection.

    Für jeden Chunk speichern wir:
      - documents: den Text selbst
      - embeddings: den Vektor
      - metadatas: Quelle + chunk_index (für Quellenangabe)
      - ids: eine eindeutige ID (Quelle + Index)
    """
    ids = [f"{c['source']}::{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"source": c["source"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query(collection, query_embedding: list[float], top_k: int) -> list[dict]:
    """
    Sucht die top_k ähnlichsten Chunks zu einem Frage-Vektor (Schritt 5-Kern).

    Rückgabe: Liste von Dicts mit Text, Quelle und Distanz (kleiner = ähnlicher).
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # ChromaDB gibt verschachtelte Listen zurück (eine pro Anfrage). Wir haben
    # nur eine Anfrage gestellt -> wir nehmen jeweils das erste Element [0].
    hits: list[dict] = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": distance,
        })
    return hits
