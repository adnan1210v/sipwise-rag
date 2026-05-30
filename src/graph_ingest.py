"""
GraphRAG-BEFÜLLUNG: baut aus den Dokumenten den Knowledge Graph in Neo4j.

Das ist das Gegenstück zu src/ingest.py (welches die Vektor-DB befüllt). Beide
nutzen ABSICHTLICH dieselben Lade- und Chunk-Funktionen – so bleiben Vektor-Welt
und Graph-Welt auf demselben Text-Stand, ohne Code zu duplizieren.

Ablauf (4 Schritte):
  [1] Dokumente laden        (document_loader – wie beim Vector-RAG)
  [2] In Chunks zerlegen     (chunker – wie beim Vector-RAG)
  [3] Pro Chunk Triples extrahieren (graph_extractor + lokales LLM)
  [4] Triples nach Neo4j schreiben  (graph_store)

Aufruf:
    python -m src.graph_ingest

WICHTIG (RAM/Zeit-Trade-off): Schritt 3 ruft das LLM EINMAL PRO CHUNK auf – der
langsamste Teil. Deshalb begrenzt config.MAX_CHUNKS_FOR_GRAPH die Anzahl, damit
der Graph in Minuten statt Stunden steht. Für den vollen Graph: in config.py
MAX_CHUNKS_FOR_GRAPH = None setzen (dann werden ALLE Chunks verarbeitet).
"""

import time

from . import config
from .document_loader import load_documents
from .chunker import chunk_documents
from .graph_extractor import extract_triples
from .graph_store import (
    setup_schema,
    reset_graph,
    write_triples,
    graph_stats,
    ping,
    close_driver,
)


def run_graph_ingestion():
    print("=" * 60)
    print("GraphRAG-Ingestion startet (Knowledge Graph bauen)")
    print("=" * 60)

    # --- Vorab-Check: Läuft Neo4j überhaupt? --------------------------------
    # Ohne DB-Verbindung wäre die teure Extraktion verschwendet -> früh prüfen.
    print("\n[0/4] Prüfe Neo4j-Verbindung ...")
    if not ping():
        print("  ❌ Keine Verbindung zu Neo4j. Läuft der Container?")
        print("     -> docker compose up -d   (dann erneut versuchen)")
        return
    print("  ✅ Neo4j ist erreichbar.")

    # --- Schritt 1: Dokumente laden -----------------------------------------
    print(f"\n[1/4] Lade Dokumente aus: {config.DATA_DIR}")
    documents = load_documents(config.DATA_DIR)
    if not documents:
        print("  ⚠️  Keine Dokumente gefunden! Lege PDFs/.txt in data/ ab.")
        return
    print(f"  -> {len(documents)} Dokument(e) geladen.")

    # --- Schritt 2: In Chunks zerlegen --------------------------------------
    print("\n[2/4] Zerlege Dokumente in Chunks ...")
    chunks = chunk_documents(documents)
    print(f"  -> {len(chunks)} Chunks insgesamt.")

    # Begrenzen, damit die Extraktion auf 8 GB RAM nicht ewig dauert.
    limit = config.MAX_CHUNKS_FOR_GRAPH
    if limit is not None and len(chunks) > limit:
        chunks = chunks[:limit]
        print(f"  -> begrenzt auf die ersten {limit} Chunks "
              f"(config.MAX_CHUNKS_FOR_GRAPH).")
        print("     (Für den vollen Graph: in config.py auf None setzen.)")

    # --- Schritt 3+4: Frischen Graph anlegen, dann Chunk für Chunk füllen ----
    # reset_graph() leert alles -> kein Vermischen mit einem alten Lauf.
    # setup_schema() stellt die Eindeutigkeits-Regel + den Index sicher.
    print("\n[3/4] Bereite Graph vor (leeren + Schema) ...")
    reset_graph()
    setup_schema()

    print(f"\n[4/4] Extrahiere Triples mit '{config.EXTRACTION_MODEL_NAME}' "
          f"und schreibe sie nach Neo4j ...")
    print("      (Das LLM wird pro Chunk aufgerufen – bitte etwas Geduld.)\n")

    start_time = time.time()
    total_triples = 0

    for i, chunk in enumerate(chunks, start=1):
        # Schritt 3: dieser Chunk -> Liste von Triples (über das LLM).
        triples = extract_triples(chunk["text"])

        # Schritt 4: Triples dieses Chunks in den Graphen schreiben.
        # (write_triples filtert intern unvollständige Triples heraus.)
        write_triples(triples, chunk["source"], chunk["chunk_index"])
        total_triples += len(triples)

        # Fortschritts-Ausgabe alle 10 Chunks (und beim letzten), damit man bei
        # einem langen Lauf sieht, dass etwas passiert – ohne die Konsole zu fluten.
        if i % 10 == 0 or i == len(chunks):
            elapsed = time.time() - start_time
            tempo = i / elapsed if elapsed > 0 else 0
            print(f"   [{i}/{len(chunks)}] Chunks verarbeitet, "
                  f"{total_triples} Triples extrahiert "
                  f"(~{tempo:.1f} Chunks/s)")

    # --- Abschluss-Statistik ------------------------------------------------
    stats = graph_stats()
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Fertig in {elapsed:.0f}s.")
    print(f"   Roh extrahierte Triples: {total_triples}")
    print(f"   Im Graph: {stats['nodes']} Knoten, "
          f"{stats['relationships']} Kanten.")
    print("   (Weniger Kanten als Triples ist NORMAL: Duplikate werden durch "
          "MERGE\n    zusammengeführt und unvollständige Triples verworfen.)")
    print("=" * 60)
    print("\n👀 Schau dir den Graph im Browser an: http://localhost:7474")
    print("   Login: neo4j / sipwise123")
    print("   Probier diese Cypher-Abfrage (zeigt 50 Beziehungen als Bild):")
    print("     MATCH p=()-[:REL]->() RETURN p LIMIT 50")

    close_driver()


if __name__ == "__main__":
    run_graph_ingestion()
