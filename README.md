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

## 🆕 Zwei Welten: Vector-RAG und GraphRAG

Dieses Projekt hat **zwei** Retrieval-Ansätze, die nebeneinander laufen. Der
Unterschied in einem Satz:

> **Vector Search findet ähnliche *Texte*. Der Graph findet verbundene *Fakten*.
> GraphRAG kombiniert beides.**

| | Vector-RAG (Basis) | GraphRAG (Erweiterung) |
|---|---|---|
| Sucht nach | **ähnlichen Textstellen** (Bedeutung) | **verbundenen Fakten** (Beziehungen) |
| Datenbank | ChromaDB (Vektoren) | Neo4j (Knoten + Kanten) |
| Stark bei | „Worum geht es ungefähr?" | „Wie hängt A mit B und C zusammen?" |
| Schwäche | Zusammenhänge über mehrere Stellen | braucht erst eine Extraktion (LLM, langsam) |

**Analogie:** Vector-RAG ist wie eine **Volltextsuche**, die dir die passendsten
Absätze gibt. Der Graph ist wie eine **Mindmap**, in der Begriffe mit Linien
verbunden sind – du kannst von „Kamailio" zu allem springen, was damit zu tun
hat. GraphRAG legt beides übereinander: erst die Mindmap fürs „Skelett" der
Zusammenhänge, dann die Absätze fürs „Fleisch" der Details.

👉 Komplette Anleitung weiter unten: [GraphRAG einrichten & nutzen](#-graphrag-knowledge-graph-einrichten--nutzen).
Die Konzepte (warum Neo4j, warum Cypher, wie Hybrid-Retrieval funktioniert,
welche Trade-offs) stehen ausführlich in **[`ERKLAERUNG.md`](ERKLAERUNG.md)**.

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
│   ├── chat.py            # interaktiver Terminal-Chat (mehrere Fragen)
│   │                      # ── GraphRAG-Erweiterung (additiv) ──
│   ├── graph_store.py     # [G4] Neo4j: Verbindung, Schema, Schreiben/Lesen
│   ├── graph_extractor.py # [G3] LLM extrahiert Triples (Knoten/Kanten) aus Text
│   ├── graph_ingest.py    # baut den Graph (Laden→Chunking→Extraktion→Neo4j)
│   ├── graph_retriever.py # [G5] Saat-Knoten finden + Beziehungen folgen (Cypher)
│   ├── hybrid_retriever.py# kombiniert Vektor- + Graph-Kontext ("Context Fusion")
│   ├── graph_pipeline.py  # answer_question_graph (hybride Antwort)
│   └── compare.py         # Demo: dieselbe Frage Vector-RAG vs. GraphRAG
├── web/
│   └── index.html         # einfache Web-Oberfläche (Textfeld + Antwort)
├── eval/
│   ├── test_questions.json
│   ├── evaluate.py            # misst Vektor-Retrieval-Qualität (Hit@k)
│   ├── graph_test_questions.json
│   └── compare_eval.py        # vergleicht Vector-RAG vs. GraphRAG (Coverage)
├── docker-compose.yml     # Neo4j-Container (lokal, RAM-gedeckelt)
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

## 🕸️ GraphRAG (Knowledge Graph) einrichten & nutzen

Die GraphRAG-Erweiterung baut aus der Doku einen **Wissensgraphen** in **Neo4j**
und kombiniert ihn beim Antworten mit der Vektor-Suche. Sie ist **additiv**: das
Vector-RAG oben funktioniert unverändert weiter, auch wenn Neo4j gar nicht läuft.

### Zusätzliche Voraussetzungen
- **Container-Runtime** für Neo4j. Hier genutzt: **Colima** (schlanke, quelloffene
  Docker-Alternative – RAM-sparsam, ideal für 8 GB).
  ```bash
  brew install colima docker docker-compose
  # docker-compose als Plugin auffindbar machen (einmalig):
  mkdir -p ~/.docker && printf '{\n  "cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]\n}\n' > ~/.docker/config.json
  ```
- Der `neo4j`-Python-Treiber ist in `requirements.txt` enthalten
  (`pip install -r requirements.txt`).

### Schritt 1 – Neo4j-Container starten
```bash
colima start --cpu 2 --memory 4   # Linux-VM für Docker (Ollama läuft NATIV daneben)
docker compose up -d              # Neo4j aus docker-compose.yml starten
docker compose ps                 # sollte "Up ... (healthy)" zeigen
```
- **Neo4j-Browser:** http://localhost:7474 — Login `neo4j` / `sipwise123`
  (Connect-URL `bolt://localhost:7687`).
- **Verbindungstest aus Python:** `python -m src.graph_store`
  → „✅ Verbindung steht".

Container-Steuerung: `docker compose stop` (anhalten, Daten bleiben),
`docker compose start` (weiter), `docker compose down` (entfernen, Daten bleiben
dank Volume), `docker compose down -v` (⚠️ löscht auch die Graph-Daten).

### Schritt 2 – Knowledge Graph bauen (Extraktion)
```bash
python -m src.graph_ingest
```
Das schickt die Text-Chunks **einzeln durchs lokale LLM** und lässt es Fakten als
**Triples** (Subjekt → Beziehung → Objekt) extrahieren, die nach Neo4j
geschrieben werden. ⏱️ **Das ist der langsamste Schritt** (ein LLM-Aufruf pro
Chunk). Über `MAX_CHUNKS_FOR_GRAPH` in `src/config.py` ist die Anzahl begrenzt
(Default 150), damit der Graph in Minuten statt Stunden steht – siehe
[Trade-offs](#-graphrag-trade-offs-ram--geschwindigkeit).

**So sieht der Graph dann im Neo4j-Browser aus** (http://localhost:7474) – diese
Cypher-Abfrage zeichnet 50 Beziehungen als Bild:
```cypher
MATCH p=()-[:REL]->() RETURN p LIMIT 50
```
Du siehst Kreise (Entitäten wie *Kamailio*, *Sipwise C5*) und Pfeile dazwischen
(Beziehungen wie *uses*, *handles*).

### Schritt 3 – Fragen mit GraphRAG beantworten
```bash
python -m src.graph_pipeline "What components does Sipwise C5 use?"
```

### Schritt 4 – Vergleich Vector-RAG vs. GraphRAG (die Interview-Demo!)
```bash
python -m src.compare "How is call data stored and which components are involved?"
```
Zeigt **dieselbe Frage** zweimal beantwortet – einmal nur Vektor, einmal hybrid –
samt der genutzten Graph-Fakten. Wähle eine Frage über **Zusammenhänge**, dann
spielt der Graph seine Stärke aus.

### Schritt 5 – Messen: Coverage-Vergleich
```bash
python -m eval.compare_eval
```
Misst pro Ansatz, wie viele erwartete Entitäten im abgerufenen Kontext landen
(Fragen in `eval/graph_test_questions.json`). Beispielergebnis dieses Projekts:

| Ansatz | Coverage |
|---|---|
| Vector-RAG | 5/10 (50 %) |
| **GraphRAG** | **7/10 (70 %)** |

→ GraphRAG liegt bei Zusammenhang-Fragen vorne (z. B. „Wie werden Call-Daten
gespeichert?": 3/3 statt 2/3).

### 🧠 GraphRAG-Trade-offs (RAM & Geschwindigkeit)
Wichtiges Interview-Thema – bewusst getroffene Kompromisse auf einem 8-GB-Mac:

- **Colima statt Docker Desktop:** Docker Desktops VM frisst spürbar RAM; Colima
  ist schlanker. Die VM bekommt **4 GB**, Neo4j selbst ist in `docker-compose.yml`
  auf **~1 GB** gedeckelt (`heap 512m` + `pagecache 512m`). So bleibt genug RAM
  fürs **native Ollama** (LLM läuft NICHT in der VM).
- **Extraktion ist der Flaschenhals:** ein LLM-Aufruf pro Chunk. Stellschrauben in
  `src/config.py`: `EXTRACTION_MODEL_NAME` (z. B. `llama3.2:1b` statt `3b` →
  ~2× schneller, gröbere Triples) und `MAX_CHUNKS_FOR_GRAPH` (Anzahl Chunks).
  Das ist im Kern eine **Quantisierungs-/Modellgröße-vs-Qualität**-Abwägung.
- **Ein Kantentyp `:REL` mit `type`-Eigenschaft** statt dynamischer Kantentypen –
  letztere bräuchten die APOC-Erweiterung (mehr RAM/Komplexität).
- **Nur 1-Hop-Traversierung** im Graph-Retrieval: direkte Nachbarn statt tiefer
  Pfade – hält den Prompt klein und das kleine LLM fokussiert.

> 💡 Hängt die Extraktion mal an einem Chunk (kommt bei kleinen LLMs vor), kann
> man den Lauf abbrechen – bereits extrahierte Triples sind in Neo4j gespeichert.

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
- **GraphRAG: „Keine Verbindung zu Neo4j":** Container läuft nicht oder noch nicht
  bereit. Prüfen mit `docker compose ps` (Status `healthy`?) und `colima status`.
  Notfalls `colima start` und `docker compose up -d` erneut.
- **GraphRAG-Extraktion hängt/dauert ewig:** kleineres `EXTRACTION_MODEL_NAME`
  (`llama3.2:1b`) und/oder kleineres `MAX_CHUNKS_FOR_GRAPH` in `src/config.py`.
  Lauf abbrechen ist sicher – bereits geschriebene Triples bleiben im Graph.
- **`docker compose` nicht gefunden:** `~/.docker/config.json` mit
  `cliPluginsExtraDirs` anlegen (siehe GraphRAG-Setup oben).

---

## Mögliche Erweiterungen (Gesprächsstoff im Interview)

- **Re-Ranking**: abgerufene Chunks mit einem Cross-Encoder nachsortieren.
- **Bessere Metriken**: Faithfulness/Answer-Relevancy via RAGAS messen.
- **Token-basiertes Chunking** statt Zeichen (z. B. mit `tiktoken`).
- **Quellen-Hervorhebung** in der Antwort (welcher Satz aus welchem Chunk).
- **Streaming-Antworten** über die API.
