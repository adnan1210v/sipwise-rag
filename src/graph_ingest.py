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

import json
import re
import time
from pathlib import Path

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

    checkpoint_path = _checkpoint_path(len(chunks))
    records = _load_checkpoint(checkpoint_path) if config.GRAPH_INGEST_RESUME else {}

    # --- Schritt 3: Triples extrahieren und als Checkpoint speichern ----------
    # Wichtig: Neo4j wird hier noch NICHT geleert. Wenn Ollama hängt oder der Lauf
    # abbricht, bleibt der bisher funktionierende Graph erhalten. Erst wenn die
    # Extraktion abgeschlossen ist, bauen wir Neo4j aus dem Checkpoint neu auf.
    print(f"\n[3/4] Extrahiere Triples mit '{config.EXTRACTION_MODEL_NAME}' ...")
    print(f"      Checkpoint: {checkpoint_path}")
    if records:
        print(f"      Resume aktiv: {len(records)} Chunk(s) bereits im Checkpoint.")
    print("      (Das LLM wird pro noch fehlendem Chunk aufgerufen.)\n")

    start_time = time.time()
    newly_extracted = 0

    for i, chunk in enumerate(chunks, start=1):
        chunk_id = _chunk_id(chunk)
        if chunk_id in records:
            triples = records[chunk_id]["triples"]
        else:
            triples = extract_triples(chunk["text"])
            record = {
                "chunk_id": chunk_id,
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "triples": triples,
            }
            _append_checkpoint(checkpoint_path, record)
            records[chunk_id] = record
            newly_extracted += 1

        # Fortschritts-Ausgabe alle 10 Chunks (und beim letzten), damit man bei
        # einem langen Lauf sieht, dass etwas passiert – ohne die Konsole zu fluten.
        if i % 10 == 0 or i == len(chunks):
            elapsed = time.time() - start_time
            tempo = i / elapsed if elapsed > 0 else 0
            total_triples = sum(len(r["triples"]) for r in records.values())
            print(f"   [{i}/{len(chunks)}] Chunks verarbeitet, "
                  f"{total_triples} Triples im Checkpoint "
                  f"(~{tempo:.1f} Chunks/s)")

    # --- Schritt 4: Neo4j aus dem vollständigen Checkpoint neu aufbauen -------
    print("\n[4/4] Schreibe Checkpoint nach Neo4j (frischer Graph) ...")
    reset_graph()
    setup_schema()

    total_triples = 0
    for record in records.values():
        triples = record["triples"]
        write_triples(triples, record["source"], record["chunk_index"])
        total_triples += len(triples)

    # --- Abschluss-Statistik ------------------------------------------------
    stats = graph_stats()
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Fertig in {elapsed:.0f}s.")
    print(f"   Neu extrahierte Chunks: {newly_extracted}")
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


def _checkpoint_path(chunk_count: int) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", config.EXTRACTION_MODEL_NAME)
    filename = f"graph_triples_{safe_model}_{chunk_count}_chunks.jsonl"
    return config.GRAPH_CHECKPOINT_DIR / filename


def _chunk_id(chunk: dict) -> str:
    return f"{chunk['source']}::{chunk['chunk_index']}"


def _load_checkpoint(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        chunk_id = record.get("chunk_id")
        if chunk_id:
            records[chunk_id] = record
    return records


def _append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_graph_ingestion()
