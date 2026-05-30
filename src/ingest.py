"""
INGEST-Skript: führt die Schritte 1–4 als EINEN Vorgang aus ("Befüllen").

Das ist der Teil, den du EINMAL ausführst (und immer dann neu, wenn du neue
Dokumente in data/ legst). Danach steht die Wissensbasis in chroma_db/ bereit
und das Beantworten von Fragen (Schritte 5–6) läuft schnell.

Aufruf:
    python -m src.ingest
"""

from . import config
from .document_loader import load_documents
from .chunker import chunk_documents
from .embeddings import embed_texts
from .vector_store import reset_collection, add_chunks


def run_ingestion():
    print("=" * 60)
    print("RAG-Ingestion startet")
    print("=" * 60)

    # --- Schritt 1: Dokumente laden -----------------------------------------
    print(f"\n[1/4] Lade Dokumente aus: {config.DATA_DIR}")
    documents = load_documents(config.DATA_DIR)
    if not documents:
        print("  ⚠️  Keine Dokumente gefunden! Lege PDFs/.txt in data/ ab.")
        return
    print(f"  -> {len(documents)} Dokument(e) geladen:")
    for doc in documents:
        print(f"     • {doc['source']} ({len(doc['text']):,} Zeichen)")

    # --- Schritt 2: In Chunks zerlegen --------------------------------------
    print("\n[2/4] Zerlege Dokumente in Chunks ...")
    chunks = chunk_documents(documents)
    print(f"  -> {len(chunks)} Chunks erzeugt "
          f"(Größe ~{config.CHUNK_SIZE} Zeichen, Overlap {config.CHUNK_OVERLAP})")

    # --- Schritt 3: Embeddings erzeugen -------------------------------------
    print(f"\n[3/4] Erzeuge Embeddings mit '{config.EMBEDDING_MODEL_NAME}' ...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    print(f"  -> {len(embeddings)} Vektoren mit je {len(embeddings[0])} Dimensionen")

    # --- Schritt 4: In ChromaDB speichern -----------------------------------
    print(f"\n[4/4] Speichere in ChromaDB: {config.CHROMA_DIR}")
    # reset_collection() löscht alte Daten -> kein Duplikat bei erneutem Lauf.
    collection = reset_collection()
    add_chunks(collection, chunks, embeddings)
    print(f"  -> {collection.count()} Chunks in Collection "
          f"'{config.COLLECTION_NAME}' gespeichert")

    print("\n✅ Fertig! Die Wissensbasis ist bereit. Jetzt kannst du Fragen stellen.")


if __name__ == "__main__":
    run_ingestion()
