"""
Die komplette RAG-PIPELINE als eine Funktion (verbindet Schritt 5 + 6).

Hier wird "Retrieval" und "Generation" zusammengeführt: Das ist die Funktion,
die sowohl die API (api.py) als auch ein Test-/Eval-Skript aufrufen. Sie gibt
nicht nur die Antwort zurück, sondern auch die QUELLEN – für Transparenz.
"""

from .retriever import retrieve
from .generator import generate_answer


def answer_question(question: str, top_k: int | None = None) -> dict:
    """
    Beantwortet eine Frage End-to-End.

    Rückgabe:
        {
          "question": "...",
          "answer": "...",
          "sources": [ {source, chunk_index, distance}, ... ]
        }
    """
    # Schritt 5: relevante Chunks holen
    chunks = retrieve(question, top_k=top_k)

    # Wenn nichts gefunden wurde (leere DB), gar nicht erst das LLM bemühen.
    if not chunks:
        return {
            "question": question,
            "answer": "Es sind keine Dokumente in der Wissensbasis. "
                      "Bitte zuerst 'python -m src.ingest' ausführen.",
            "sources": [],
        }

    # Schritt 6: Antwort generieren
    answer = generate_answer(question, chunks)

    # Quellen kompakt zurückgeben (ohne den vollen Chunk-Text, nur Belege).
    sources = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "distance": round(c["distance"], 4),
        }
        for c in chunks
    ]

    return {"question": question, "answer": answer, "sources": sources}
