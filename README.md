# Sipwise C5 — Lokales RAG-System

Ein vollständig **lokales, offline** RAG-System (Retrieval-Augmented Generation),
das Fragen über die öffentliche **Sipwise C5 CE**-Dokumentation beantwortet.
Gebaut als Lern- und Portfolio-Projekt – datenschutzbewusst, ohne Cloud-APIs,
lauffähig auf einem **MacBook mit Apple Silicon und nur 8 GB RAM**.

> 📖 Du willst jedes Detail verstehen? Die Datei **[`ERKLAERUNG.md`](ERKLAERUNG.md)**
> erklärt das gesamte Projekt Zeile für Zeile, mit Glossar und typischen
> Interview-Fragen. Du kannst sie auch Claude geben und gezielt nachfragen.

---

## Was ist RAG (in einem Satz)?

Statt das LLM frei antworten zu lassen, **suchen** wir zuerst die passenden
Textstellen aus echten Dokumenten heraus (**Retrieval**) und lassen das LLM die
Antwort nur **auf Basis dieser Stellen formulieren** (**Generation**). Ergebnis:
faktentreue Antworten mit Quellenangabe statt Halluzinationen.

---

## Architektur

```
                          EINMALIGES BEFÜLLEN (python -m src.ingest)
   data/*.pdf  ─▶ [1] Laden ─▶ [2] Chunking ─▶ [3] Embeddings ─▶ [4] ChromaDB
                                                                       │
                          FRAGE BEANTWORTEN (API / CLI)                │ speichert
   Frage ─▶ [5] Retrieval ◀───────────────────────────────────────────┘
                  │ relevante Chunks
                  ▼
            [6] LLM (Ollama)  ─▶  Antwort + Quellen
```

---

## Technischer Stack & Designentscheidungen

| Baustein | Wahl | Warum (für 8 GB RAM, offline) |
|---|---|---|
| Sprache/Umgebung | **Python 3.12 + venv** | 3.12 hat stabile, vorgebaute Pakete für PyTorch/ChromaDB (das neuere 3.14 oft noch nicht). venv isoliert die Abhängigkeiten. |
| LLM | **Ollama + `llama3.2:3b`** (~2 GB) | Läuft lokal in eigenem Prozess, lädt nur bei Bedarf. Klein genug für 8 GB. Bei Speicherdruck: in `src/config.py` auf `llama3.2:1b` umstellen. |
| Embeddings | **`all-MiniLM-L6-v2`** (~90 MB, 384 Dim.) | Winzig, schnell, offline; gut für englische technische Texte. |
| Vektor-DB | **ChromaDB (PersistentClient)** | Lokal, kein Server, speichert auf Platte. Einfache API, ideal zum Lernen. |
| Chunking | eigener Splitter, **800 Zeichen / 120 Overlap** | Nachvollziehbar (selbst gebaut), trennt an Absatz-/Satzgrenzen. Balance aus Präzision und Kontext. |
| API + UI | **FastAPI + uvicorn** | Schlank, automatische Doku unter `/docs`, einfache Web-Oberfläche unter `/`, produktnah. |

---

## Projektstruktur

```
sipwise-rag/
├── data/                  # HIER legst du deine PDFs/.txt ab
├── chroma_db/             # Vektor-DB (wird automatisch erzeugt)
├── src/
│   ├── config.py          # alle Einstellungen an einem Ort
│   ├── document_loader.py # [1] Dokumente laden
│   ├── chunker.py         # [2] Text in Chunks zerlegen
│   ├── embeddings.py      # [3] Embeddings erzeugen
│   ├── vector_store.py    # [4] speichern + suchen (ChromaDB)
│   ├── ingest.py          # führt Schritte 1–4 aus (Befüllen)
│   ├── retriever.py       # [5] relevante Chunks abrufen
│   ├── generator.py       # [6] Antwort vom LLM generieren
│   ├── pipeline.py        # verbindet 5+6 (answer_question)
│   ├── api.py             # [7] FastAPI-Endpoint + Web-Oberfläche
│   ├── ask.py             # CLI für genau eine Frage
│   └── chat.py            # interaktiver Terminal-Chat (mehrere Fragen)
├── web/
│   └── index.html         # einfache Web-Oberfläche (Textfeld + Antwort)
├── eval/
│   ├── test_questions.json
│   └── evaluate.py        # misst Retrieval-Qualität (Hit@k)
├── requirements.txt
├── README.md
└── ERKLAERUNG.md          # ausführliche Erklärung zum Lernen
```

---

## Setup (Schritt für Schritt)

### 1. Ollama-Modell laden (einmalig, braucht Internet)
```bash
ollama pull llama3.2:3b
```
Stelle sicher, dass der Ollama-Dienst läuft (die Mac-App starten oder `ollama serve`).

### 2. Virtuelle Umgebung mit Python 3.12 erstellen
```bash
cd sipwise-rag
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Abhängigkeiten installieren
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Dokumente ablegen
Lege deine Dateien in den Ordner **`data/`** (PDF, `.txt` oder `.md`),
z. B. das Sipwise C5 CE Handbook als PDF.

### 5. Wissensbasis befüllen (Schritte 1–4)
```bash
python -m src.ingest
```
Beim ersten Lauf lädt `sentence-transformers` einmal das Embedding-Modell herunter.

### 6. Fragen stellen

Es gibt drei Wege – such dir aus, was dir gefällt:

**Variante A – Interaktiver Terminal-Chat (am einfachsten):**
```bash
python -m src.chat
```
Dann einfach Fragen eintippen (ohne Anführungszeichen), Enter drücken, beliebig
oft. Beenden mit `exit` oder `Strg+C`.

**Variante B – Web-Oberfläche (hübsch, im Browser):**
```bash
uvicorn src.api:app --reload
```
Dann im Browser öffnen: **`http://127.0.0.1:8000/`** – ein Textfeld zum Fragen
stellen, Antwort schön formatiert mit Quellen.
*(Hinweis: Standard-Port ist 8000. Ist er belegt, mit `--port 8001` starten und
dann `http://127.0.0.1:8001/` öffnen.)*

**Variante C – Genau eine Frage von der Kommandozeile:**
```bash
python -m src.ask "How do I configure SIP peering?"
```

> Für Entwickler: Die rohe JSON-API liegt unter `POST /query`, die automatische
> API-Doku unter `http://127.0.0.1:8000/docs`. Direkt testen mit curl:
> ```bash
> curl -X POST http://127.0.0.1:8000/query \
>      -H "Content-Type: application/json" \
>      -d '{"question": "What is Sipwise C5?"}'
> ```

### 7. Qualität messen
```bash
python -m eval.evaluate
```
Gibt einen **Hit@k**-Wert aus (fanden wir für die Testfragen die richtige Quelle?).

---

## Tipps & Troubleshooting

- **Speicher wird knapp / sehr langsam:** In `src/config.py` `LLM_MODEL_NAME` auf
  ein kleineres Modell setzen, z. B. `"llama3.2:1b"` oder `"qwen2.5:1.5b"`.
  Optional `TOP_K` auf 3 reduzieren.
- **„connection refused" bei Ollama:** Der Ollama-Dienst läuft nicht – Mac-App
  starten oder `ollama serve` in einem zweiten Terminal.
- **Neue/geänderte Dokumente:** Einfach `python -m src.ingest` erneut ausführen
  (die DB wird dabei sauber neu aufgebaut).
- **Reproduzierbarkeit (gut fürs Interview):** Nach der Installation
  `pip freeze > requirements.lock.txt`.

---

## Mögliche Erweiterungen (Gesprächsstoff im Interview)

- **Re-Ranking**: abgerufene Chunks mit einem Cross-Encoder nachsortieren.
- **Bessere Metriken**: Faithfulness/Answer-Relevancy via RAGAS messen.
- **Token-basiertes Chunking** statt Zeichen (z. B. mit `tiktoken`).
- **Quellen-Hervorhebung** in der Antwort (welcher Satz aus welchem Chunk).
- **Streaming-Antworten** über die API.
