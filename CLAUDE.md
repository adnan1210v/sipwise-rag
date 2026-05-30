# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Tiefe Erklärungen stehen in **`ERKLAERUNG.md`**, Setup/Bedienung in **`README.md`**.

## Was ist das?
Ein **lokales, offline RAG-System** (Retrieval-Augmented Generation), das Fragen
über die **Sipwise C5 CE**-Dokumentation (VoIP-Handbuch + Fact-Sheets als PDF)
beantwortet. Portfolio-/Lernprojekt für ein Bewerbungsgespräch. Läuft komplett
offline auf einem **MacBook (Apple Silicon, 8 GB RAM)** – kein Cloud-API.

## Stack
- **Python 3.12** in venv (`.venv/`)
- **LLM:** Ollama mit `llama3.2:3b` (umstellbar in `src/config.py`)
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers` (384 Dim.)
- **Vektor-DB:** ChromaDB (`PersistentClient`, speichert in `chroma_db/`)
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

## Architektur (das große Bild)
Die Pipeline besteht aus 6 Schritten, je ein Modul in `src/`:
1. **Laden** (`document_loader.py`) – PDFs aus `data/` → Text
2. **Chunking** (`chunker.py`) – 800 Zeichen/120 Overlap (aus `config.py`)
3. **Embeddings** (`embeddings.py`) – Text → 384-dim Vektoren
4. **Speichern** (`vector_store.py`) – ChromaDB-Collection `sipwise_docs`
5. **Retrieval** (`retriever.py`) – Frage einbetten, Top-K=4 ähnlichste Chunks
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
- `generator.py` baut den Prompt mit nummerierten `[Quelle N: <Datei>]`-Blöcken,
  damit das LLM Quellen zitieren kann; der **System-Prompt** kommt aus
  `config.py` (nicht im Generator hartkodiert).

## Konventionen / Hinweise
- Alle „Stellschrauben" (Modelle, Pfade, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`,
  `TEMPERATURE`, `MAX_TOKENS`, `SYSTEM_PROMPT`) liegen zentral in `src/config.py`.
  Änderungen am RAG-Verhalten gehören dorthin, nicht in die Module.
- **Imports laufen über das Paket** (`from src.xyz import ...`); daher alles als
  Modul starten (`python -m src.chat`), nicht direkt (`python src/chat.py`).
- Code-Kommentare und Doku sind auf **Deutsch**, der Nutzer ist Anfänger und
  möchte alles verstehen und im Interview erklären können → Antworten einfach
  halten und Designentscheidungen begründen.
- Der System-Prompt in `config.py` erlaubt Vereinfachen, verbietet aber
  erfundene Fakten/Abkürzungen; bei fehlendem Kontext soll das LLM ehrlich
  „steht nicht in der Doku" sagen.
