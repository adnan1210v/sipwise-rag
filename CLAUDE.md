# Projekt-Kontext für KI-Assistenten

> Diese Datei wird von Claude Code automatisch beim Öffnen des Projekts gelesen.
> Für andere KIs (Claude.ai, ChatGPT) einfach **diese Datei** in den Chat einfügen.
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

## Pipeline (6 Schritte)
1. Laden (`document_loader.py`) → 2. Chunking (`chunker.py`, 800 Zeichen/120 Overlap)
→ 3. Embeddings (`embeddings.py`) → 4. Speichern (`vector_store.py`, ChromaDB)
→ 5. Retrieval (`retriever.py`, Top-K=4) → 6. Generation (`generator.py`, Ollama).
`pipeline.py` verbindet 5+6 zu `answer_question()`.

## Wichtige Befehle
```bash
source .venv/bin/activate          # Umgebung aktivieren (immer zuerst)
python -m src.ingest               # DB aus data/ befüllen (Schritte 1–4)
python -m src.chat                 # interaktiver Terminal-Chat
python -m src.ask "Frage"          # genau eine Frage
uvicorn src.api:app --reload       # Web-UI unter http://127.0.0.1:8000/
python -m eval.evaluate            # Retrieval-Qualität messen (Hit@k)
```

## Struktur (Kurzform)
- `src/` – Pipeline-Module + `api.py` (FastAPI), `ask.py`, `chat.py`
- `web/index.html` – einfache Web-Oberfläche
- `eval/` – Testfragen + `evaluate.py`
- `data/` – die PDFs (vom Nutzer abgelegt), `chroma_db/` – generierte Vektor-DB

## Konventionen / Hinweise
- Alle „Stellschrauben" (Modelle, Pfade, Chunk-Größe, TOP_K) liegen zentral in
  `src/config.py`.
- Code-Kommentare und Doku sind auf **Deutsch**, der Nutzer ist Anfänger und
  möchte alles verstehen und im Interview erklären können → Antworten einfach
  halten und Designentscheidungen begründen.
- Der System-Prompt in `generator.py` erlaubt Vereinfachen, verbietet aber
  erfundene Fakten/Abkürzungen.
