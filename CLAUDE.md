# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Tiefe Erklärungen stehen in **`ERKLAERUNG.md`**, Setup/Bedienung in **`README.md`**.

## Was ist das?
Ein **lokales, offline RAG-System** (Retrieval-Augmented Generation), das Fragen
über die **Sipwise C5 CE**-Dokumentation (VoIP-Handbuch + Fact-Sheets als PDF)
beantwortet. Portfolio-/Lernprojekt für ein Bewerbungsgespräch. Läuft komplett
offline auf einem **MacBook (Apple Silicon, 8 GB RAM)** – kein Cloud-API.

Es gibt **zwei** Retrieval-Ansätze nebeneinander: das **Vector-RAG** (Basis) und
die additive **GraphRAG**-Erweiterung (Neo4j + hybrides Retrieval). GraphRAG ist
rein additiv – das Vector-RAG läuft unverändert, auch ohne Neo4j.

## Stack
- **Python 3.12** in venv (`.venv/`)
- **LLM:** Ollama mit `llama3.2:3b` (umstellbar in `src/config.py`)
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers` (384 Dim.)
- **Vektor-DB:** ChromaDB (`PersistentClient`, speichert in `chroma_db/`)
- **Graph-DB (GraphRAG):** Neo4j 5.26 via Docker/Colima (`docker-compose.yml`)
- **API/UI:** FastAPI + uvicorn

## Voraussetzungen (vor dem ersten Start)
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama serve            # Ollama muss laufen (Standard: http://localhost:11434)
ollama pull llama3.2:3b # einmalig das LLM-Modell laden
```
Ohne laufendes Ollama schlägt jede **Generation** fehl (`generator.py` ruft
`POST {OLLAMA_HOST}/api/generate`). Das **Retrieval** allein (z. B. `eval`)
braucht kein Ollama, nur eine befüllte `chroma_db/`.

## Wichtige Befehle
```bash
source .venv/bin/activate          # Umgebung aktivieren (immer zuerst)
python -m src.ingest               # ZUERST: DB aus data/ befüllen (Schritte 1–4)
python -m src.chat                 # interaktiver Terminal-Chat
python -m src.ask "Frage"          # genau eine Frage
uvicorn src.api:app --reload       # Web-UI unter http://127.0.0.1:8000/
python -m eval.evaluate            # Retrieval-Qualität messen (Hit@k)
```
**Reihenfolge zählt:** `chat`/`ask`/`api`/`evaluate` setzen eine befüllte
`chroma_db/` voraus → vorher immer `python -m src.ingest` laufen lassen.
(Es gibt keine Test-Suite und kein Lint-Setup; `eval/evaluate.py` ist das
Qualitätsmaß für das Retrieval.)

### GraphRAG-Befehle (additive Erweiterung)
```bash
colima start --cpu 2 --memory 4    # Docker-VM (Ollama läuft NATIV daneben, nicht drin)
docker compose up -d               # Neo4j-Container starten (docker-compose.yml)
python -m src.graph_store          # Verbindungs-Selbsttest (Neo4j erreichbar?)
python -m src.graph_ingest         # Graph bauen: Chunks → LLM-Triples → Neo4j (LANGSAM)
python -m src.graph_pipeline "Frage"   # hybride Antwort (Vektor + Graph)
python -m src.compare "Frage"      # Demo: Vector-RAG vs. GraphRAG nebeneinander
python -m eval.compare_eval        # Coverage-Vergleich Vector vs. GraphRAG
```
**Reihenfolge GraphRAG:** Neo4j starten → `graph_ingest` (setzt befüllte
`chroma_db/` für die Chunks *nicht* voraus, lädt selbst aus `data/`; nutzt aber
Ollama) → dann `graph_pipeline`/`compare`/`compare_eval`. Neo4j-Browser:
http://localhost:7474 (Login `neo4j`/`sipwise123`, Bolt-URL `bolt://localhost:7687`).

## Architektur (das große Bild)
Die Pipeline besteht aus 6 Schritten, je ein Modul in `src/`:
1. **Laden** (`document_loader.py`) – PDFs aus `data/` → Text
2. **Chunking** (`chunker.py`) – 800 Zeichen/120 Overlap (aus `config.py`)
3. **Embeddings** (`embeddings.py`) – Text → 384-dim Vektoren
4. **Speichern** (`vector_store.py`) – ChromaDB-Collection `sipwise_docs`
5. **Retrieval** (`retriever.py`) – Frage einbetten, Top-K=4 ähnlichste Chunks
   (`query_expander.py` erzeugt vorher Deutsch/Englisch-Suchvarianten +
   Tippfehler-Korrekturen)
6. **Generation** (`generator.py`) – Chunks + Frage → Ollama → Antwort

**Zwei Phasen, klar getrennt:**
- **Ingest** (`ingest.py`) führt Schritte 1–4 offline aus und schreibt nach
  `chroma_db/`. Einmalig bzw. bei neuen PDFs.
- **Query** (`pipeline.py::answer_question`) verbindet Schritte 5+6 und ist der
  *einzige* gemeinsame Einstiegspunkt für `chat.py`, `ask.py` und `api.py`.
  Rückgabe ist immer ein dict `{"answer": str, "sources": list[dict]}`; jeder
  Source-Chunk hat `text`, `metadata` (u. a. `source` = Dateiname) und `distance`.

**Wichtige Implementierungs-Details (mehrere Dateien nötig zum Verstehen):**
- `vector_store.py` cached Client *und* Collection als Modul-Globals
  (`_client`, `_collection`) – Singleton-Muster, damit ChromaDB nicht mehrfach
  geöffnet wird. `reset_collection()` löscht für sauberes Neu-Einspielen.
- Collection nutzt **Cosine-Distanz** (`metadata={"hnsw:space": "cosine"}`) –
  Embedding-Modell und Distanzmaß müssen zusammenpassen.
- `query_expander.py` verbessert deutsche Fragen gegen englische Doku ohne neue
  DB: Originalfrage + korrigierte deutsche Frage + englische Suchvariante werden
  gesucht, Treffer dedupliziert und wieder auf `TOP_K` gekürzt.
- `generator.py` baut den Prompt mit nummerierten `[Quelle N: <Datei>]`-Blöcken,
  gibt normalisierte Suchvarianten als Hinweis mit und setzt die Antwortsprache
  explizit auf Deutsch oder Englisch.

## GraphRAG-Architektur (additive Erweiterung)
Parallel zur Vektor-Pipeline, gleiche „Form" der Einstiegspunkte:
- **Bauen:** `graph_ingest.py` nutzt *dieselben* `document_loader`/`chunker` wie
  das Vector-RAG, schickt Chunks durch `graph_extractor.py` (LLM → JSON-Triples)
  und schreibt sie via `graph_store.py` nach Neo4j.
- **Abfragen:** `graph_retriever.py` findet Saat-Knoten (Stichwort-`CONTAINS`) und
  folgt deren Kanten (1-Hop). `hybrid_retriever.py` kombiniert Vektor-Chunks +
  Graph-Fakten zu *einem* Kontext mit zwei getrennten Abschnitten („Context
  Fusion"). `graph_pipeline.py::answer_question_graph` ist das hybride Pendant zu
  `pipeline.py::answer_question` (gleiche Rückgabe-Form + `graph_facts`/`seed_entities`).

**Wichtige Details:**
- **Datenmodell:** `(:Entity {name, display, type})-[:REL {type, source_doc,
  chunk_index}]->(:Entity)`. Bewusst EIN Kantentyp `:REL` mit `type`-Eigenschaft
  (dynamische Kantentypen bräuchten APOC). `name` ist normalisiert (lowercase) →
  Dedup via `MERGE`; Constraint `entity_name_unique` erzwingt Eindeutigkeit + Index.
- `graph_store.py` cached den Treiber als Modul-Global (`_driver`, Singleton wie
  bei ChromaDB/Embeddings). Lesende Queries über `run_read(cypher, **params)`
  (parametrisiert → kein Cypher-Injection).
- **Robustheit:** `write_triples` verwirft unvollständige Triples;
  `graph_extractor._parse_triples` heilt kaputtes LLM-JSON (schneidet `[`…`]`).
  `hybrid_retrieve` fängt Neo4j-Fehler ab → fällt auf reines Vector-RAG zurück.
- **Cypher steht direkt im Code, ausführlich kommentiert** (Lernziel) – v. a. in
  `graph_store.py` und `graph_retriever.py`.

**GraphRAG-Stellschrauben in `config.py`:** `NEO4J_URI/USER/PASSWORD` (per Env
überschreibbar), `EXTRACTION_MODEL_NAME` (3b↔1b: RAM/Tempo vs. Qualität),
`MAX_CHUNKS_FOR_GRAPH` (Default 150; `None` = alle), `GRAPH_SEED_ENTITIES`,
`GRAPH_MAX_FACTS`.

## Konventionen / Hinweise
- Alle „Stellschrauben" (Modelle, Pfade, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`,
  `QUERY_EXPANSION_ENABLED`, `QUERY_EXPANSION_MAX_VARIANTS`) liegen zentral in `src/config.py`.
  Änderungen am RAG-Verhalten gehören dorthin, nicht in die Module.
- **Imports laufen über das Paket** (`from src.xyz import ...`); daher alles als
  Modul starten (`python -m src.chat`), nicht direkt (`python src/chat.py`).
- Code-Kommentare und Doku sind auf **Deutsch**, der Nutzer ist Anfänger und
  möchte alles verstehen und im Interview erklären können → Antworten einfach
  halten und Designentscheidungen begründen.
- Der System-Prompt in `config.py` erlaubt Vereinfachen, verbietet aber
  erfundene Fakten/Abkürzungen; bei fehlendem Kontext soll das LLM ehrlich
  „steht nicht in der Doku" sagen.
