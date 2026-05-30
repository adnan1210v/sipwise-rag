# 📖 ERKLAERUNG — Das RAG-System vollständig verstehen

Dieses Dokument erklärt das gesamte Projekt von Grund auf. Es ist so geschrieben,
dass du es **Schritt für Schritt durchlesen** kannst, ohne Vorwissen. Du kannst
es auch Claude (oder einem anderen Assistenten) geben und gezielt nachfragen,
z. B. *„Erklär mir Abschnitt 4 nochmal mit einer Analogie"*.

**Inhalt**
1. [Das große Bild: Was ist RAG und warum?](#1-das-große-bild)
2. [Grundbegriffe (Glossar)](#2-grundbegriffe-glossar)
3. [Die Pipeline im Detail – Schritt für Schritt](#3-die-pipeline-im-detail)
4. [Embeddings & Ähnlichkeit anschaulich erklärt](#4-embeddings--ähnlichkeit)
5. [Warum diese Designentscheidungen?](#5-warum-diese-designentscheidungen)
6. [Datei-für-Datei-Führung durch den Code](#6-datei-für-datei)
7. [Ein kompletter Durchlauf an einem Beispiel](#7-ein-kompletter-durchlauf)
8. [Evaluation: Woher weiß ich, dass es funktioniert?](#8-evaluation)
9. [Typische Interview-Fragen & gute Antworten](#9-interview-fragen)
10. [Häufige Fehler & Lösungen](#10-häufige-fehler)

---

## 1. Das große Bild

### Das Problem
Ein Sprachmodell (LLM) wie Llama kennt nur das, was es im Training gesehen hat.
Über die **interne Sipwise-Doku** weiß es wenig – und wenn man trotzdem fragt,
**erfindet** es oft plausibel klingende, aber falsche Antworten
(„Halluzination"). Außerdem will man die Doku nicht in eine Cloud schicken
(Datenschutz).

### Die Idee von RAG
**R**etrieval-**A**ugmented **G**eneration heißt: Wir **erweitern** (augment) die
Antwortgenerierung um einen **Such**-Schritt (retrieval).

> **Analogie:** Ein LLM ohne RAG ist wie ein Student, der eine Klausur aus dem
> Gedächtnis schreibt. RAG ist eine **Open-Book-Klausur**: Der Student darf
> vorher die relevanten Seiten heraussuchen und nur daraus antworten.

Zwei Phasen:

| Phase | Wann | Was passiert |
|---|---|---|
| **Indexieren** (Ingestion) | einmalig | Dokumente werden in durchsuchbare „Häppchen" verwandelt und in einer Vektor-DB abgelegt. |
| **Abfragen** (Query) | bei jeder Frage | Passende Häppchen werden gefunden und dem LLM als Kontext mitgegeben. |

---

## 2. Grundbegriffe (Glossar)

- **LLM (Large Language Model):** Das Modell, das Text generiert (hier Llama 3.2
  über Ollama). „Generiert" = sagt das nächste Wort vorher, immer wieder.
- **Token:** Ein Wort-Stück. LLMs rechnen nicht in Buchstaben, sondern in Tokens
  (z. B. „peering" ≈ 2 Tokens). Modelle haben ein **Kontextfenster** = wie viele
  Tokens sie auf einmal lesen können.
- **Embedding:** Eine Liste von Zahlen (ein **Vektor**), die die **Bedeutung**
  eines Textes darstellt. Ähnliche Bedeutung → ähnliche Zahlen.
- **Vektor-Datenbank:** Eine DB, die solche Zahlen-Vektoren speichert und sehr
  schnell die **ähnlichsten** finden kann (hier ChromaDB).
- **Chunk:** Ein kleines Textstück eines Dokuments (hier ~800 Zeichen).
- **Retrieval:** Der Such-Schritt — die relevantesten Chunks zu einer Frage holen.
- **Cosine-Distanz:** Ein Maß dafür, wie „ähnlich" zwei Vektoren zeigen
  (kleiner = ähnlicher). Mehr dazu in Abschnitt 4.
- **Ollama:** Ein Programm, das LLMs lokal auf deinem Mac ausführt (kein Internet
  nötig, nachdem das Modell geladen ist).
- **Halluzination:** Wenn ein LLM etwas Falsches selbstbewusst erfindet. RAG
  reduziert das, indem es das Modell an echte Quellen bindet.

---

## 3. Die Pipeline im Detail

Es gibt **6 Pipeline-Schritte** (+ API als Schritt 7). Jeder Schritt lebt in
einer eigenen Datei – so bleibt alles übersichtlich und einzeln testbar.

### Indexieren (einmalig, `python -m src.ingest`)

**[1] Laden** — `document_loader.py`
> Liest alle PDFs/Texte aus `data/`. Aus jedem PDF wird der reine Text
> extrahiert. Der **Dateiname** wird als *Quelle* gemerkt.
> *Warum die Quelle merken?* Damit jede spätere Antwort belegen kann, woher sie
> stammt – das ist der Vertrauensgewinn von RAG.

**[2] Chunking** — `chunker.py`
> Der lange Text wird in überlappende Stücke (~800 Zeichen) geschnitten.
> *Warum?* (a) Das Embedding-Modell kann nur kurze Texte verarbeiten. (b) Wir
> wollen **präzise** die richtige Stelle finden, nicht das ganze Buch. (c) Das
> LLM soll nur wenige relevante Häppchen lesen (spart RAM/Zeit).
> *Warum Overlap?* Damit ein Satz an einer Schnittgrenze nicht zerrissen wird.

**[3] Embeddings** — `embeddings.py`
> Jeder Chunk wird in einen 384-Zahlen-Vektor umgewandelt. Das macht das Modell
> `all-MiniLM-L6-v2`. Jetzt ist „Bedeutung" mathematisch vergleichbar.

**[4] Speichern** — `vector_store.py`
> Chunks + Vektoren + Metadaten (Quelle, Index) landen in ChromaDB auf der
> Festplatte (`chroma_db/`). Das ist unsere durchsuchbare Wissensbasis.

### Abfragen (bei jeder Frage)

**[5] Retrieval** — `retriever.py`
> Die Frage wird mit **demselben** Embedding-Modell in einen Vektor verwandelt.
> ChromaDB findet die `TOP_K` (=4) ähnlichsten Chunks.

**[6] Generation** — `generator.py`
> Die gefundenen Chunks + die Frage werden zu einem **Prompt** zusammengebaut und
> an das lokale LLM (Ollama) geschickt. Ein **System-Prompt** weist es an, NUR
> aus dem Kontext zu antworten und nichts zu erfinden.

**[7] API/CLI** — `api.py` / `ask.py`
> Stellt das Ganze als Web-Endpoint bzw. Kommandozeilen-Tool bereit.

`pipeline.py` verbindet Schritt 5 + 6 zur Funktion `answer_question()`, die
sowohl die API als auch das Eval-Skript benutzen.

---

## 4. Embeddings & Ähnlichkeit

Das ist das Herzstück und für Anfänger oft das Magische. Hier die Intuition.

### Bedeutung als Punkt im Raum
Stell dir eine Landkarte vor. Jeder Text bekommt eine Position. Texte mit
ähnlicher Bedeutung liegen **nah beieinander**:

```
        • "configure SIP peering"
       • "set up peering for SIP"        ← liegen nah (ähnliche Bedeutung)
                                   
                          • "billing rates"   ← liegt weit weg (anderes Thema)
```

Statt 2 Dimensionen (Landkarte) benutzt `all-MiniLM-L6-v2` **384 Dimensionen** –
für uns nicht vorstellbar, aber mathematisch genau dasselbe Prinzip.

### Wie misst man „nah"?
Mit der **Cosine-Ähnlichkeit**: Sie schaut auf den **Winkel** zwischen zwei
Vektoren, nicht auf ihre Länge.
- Zeigen beide in dieselbe Richtung → Winkel ~0° → sehr ähnlich.
- Stehen sie senkrecht → unabhängig.
ChromaDB benutzt die **Cosine-Distanz** = quasi „1 minus Ähnlichkeit", also:
**kleinere Distanz = ähnlicher**. Deshalb sortieren wir aufsteigend nach Distanz.

### Warum dasselbe Modell für Frage und Dokumente?
Frage und Chunks müssen auf **dieselbe Landkarte** gezeichnet werden, sonst sind
die Positionen nicht vergleichbar. Deshalb nutzt `embed_query()` exakt das
Modell aus `embeddings.get_model()`.

---

## 5. Warum diese Designentscheidungen?

| Entscheidung | Alternative | Warum so gewählt |
|---|---|---|
| **Python 3.12** | 3.14 (Systemstandard) | 3.14 ist sehr neu; PyTorch/ChromaDB haben dafür oft noch keine vorgebauten Pakete → Installationsfehler. 3.12 ist stabil. |
| **`llama3.2:3b`** | 1b / 7b+ | 7b sprengt 8 GB RAM. 1b ist schneller, aber ungenauer. 3b ist der beste Kompromiss; per Config in 1 Zeile auf 1b umstellbar. |
| **`all-MiniLM-L6-v2`** | größere Embedder | Größere Embedder sind genauer, aber langsamer/größer. MiniLM ist winzig und für unsere Doku gut genug. |
| **ChromaDB** | FAISS, Pinecone, pgvector | FAISS hat keine Metadaten/Persistenz out-of-the-box. Pinecone ist Cloud (verboten: offline!). ChromaDB ist lokal, einfach, mit Metadaten. |
| **800 Zeichen Chunks** | 200 / 2000 | 200 → Kontext zerfällt; 2000 → unpräzise Treffer + mehr RAM. 800 ≈ 1–2 Absätze. |
| **Eigener Chunker** | LangChain-Splitter | Bewusst selbst gebaut, damit du jede Zeile verstehst (Lernziel). LangChain wäre die „Profi-Abkürzung". |
| **`temperature=0.1`** | höher | Niedrige Temperatur → faktentreue, wenig „kreative" Antworten. Genau was eine Doku-QA braucht. |

---

## 6. Datei-für-Datei

> Tipp: Öffne die jeweilige Datei parallel – die Kommentare im Code ergänzen das
> hier Gesagte.

- **`src/config.py`** — Alle Stellschrauben (Modelle, Pfade, Chunk-Größe, TOP_K)
  an EINEM Ort. Erste Anlaufstelle für Experimente.
- **`src/document_loader.py`** — `load_documents()` liest PDFs (via `pypdf`) und
  Textdateien, überspringt Unbekanntes und Leeres, gibt `{"source", "text"}` zurück.
- **`src/chunker.py`** — `chunk_documents()` zerschneidet Texte mit Overlap und
  versucht, an Absatz-/Satzgrenzen zu trennen. Hängt Metadaten (`source`,
  `chunk_index`) an.
- **`src/embeddings.py`** — Lädt das Embedding-Modell EINMAL (Singleton).
  `embed_texts()` für viele Chunks, `embed_query()` für die Frage.
- **`src/vector_store.py`** — Kapselt ChromaDB: `get_collection()`,
  `reset_collection()` (verhindert Duplikate), `add_chunks()`, `query()`.
- **`src/ingest.py`** — Orchestriert Schritte 1–4 mit Fortschritts-Ausgaben.
  Das ist dein „Befüll"-Befehl.
- **`src/retriever.py`** — `retrieve()`: Frage → Vektor → ähnlichste Chunks.
- **`src/generator.py`** — Baut den Prompt (System-Prompt + nummerierter Kontext
  + Frage), ruft `ollama.chat()`. Der System-Prompt ist die „Anti-Halluzinations-
  Versicherung": Er erlaubt dem Modell ausdrücklich, den Kontext zu **vereinfachen
  und in einfachen Worten zu erklären** (z. B. „für einen Anfänger"), verbietet
  aber, Fakten/Abkürzungen zu **erfinden**, und bittet um Antwort in der Sprache
  der Frage. (Frühe Version war zu streng → das Modell verweigerte Vereinfachungen.)
- **`src/pipeline.py`** — `answer_question()`: verbindet Retrieval + Generation,
  liefert Antwort **und** Quellen.
- **`src/api.py`** — FastAPI mit drei Routen: `/` (liefert die Web-Oberfläche),
  `/health` (Statuscheck) und `/query` (die eigentliche Frage-API). Pydantic-
  Modelle validieren Ein-/Ausgaben automatisch. Auto-Doku unter `/docs`.
- **`web/index.html`** — Die einfache Web-Oberfläche (ein Textfeld + Antwortbereich).
  Reines HTML/JS in einer Datei, ohne Framework → offline lauffähig. Per `fetch()`
  ruft sie genau denselben `/query`-Endpoint auf wie curl, zeigt die Antwort aber
  schön formatiert (CSS `white-space: pre-wrap` macht aus den `\n` echte Absätze).
- **`src/ask.py`** — CLI für genau **eine** Frage (Frage als Argument).
- **`src/chat.py`** — Interaktiver Terminal-Chat: einmal starten, dann beliebig
  viele Fragen tippen (eine `while`-Schleife mit `input()`). Beenden mit `exit`.
- **`eval/`** — Testfragen + `evaluate.py` (misst Hit@k).

---

## 7. Ein kompletter Durchlauf

Angenommen, du fragst: **„How do I configure SIP peering?"**

1. **`ask.py`** ruft `answer_question("How do I configure SIP peering?")`.
2. **`retriever.retrieve()`**
   - `embed_query()` macht aus der Frage einen 384-Zahlen-Vektor.
   - `vector_store.query()` fragt ChromaDB nach den 4 ähnlichsten Chunks.
   - Ergebnis z. B.: 4 Chunks aus `handbook.pdf` mit Distanzen 0.21–0.38.
3. **`generator.generate_answer()`**
   - Baut den Prompt:
     ```
     [Kontext 1 | Quelle: handbook.pdf] ...Text über Peering...
     [Kontext 2 | Quelle: handbook.pdf] ...
     ...
     Frage: How do I configure SIP peering?
     Antwort (nur basierend auf dem Kontext oben):
     ```
   - Schickt das an `llama3.2:3b` über Ollama (lokal).
   - Das LLM formuliert eine Antwort NUR aus diesen Chunks.
4. **`pipeline.py`** gibt zurück:
   ```json
   {
     "question": "How do I configure SIP peering?",
     "answer": "To configure SIP peering, ...",
     "sources": [
       {"source": "handbook.pdf", "chunk_index": 142, "distance": 0.21},
       ...
     ]
   }
   ```
5. **`ask.py`** zeigt Antwort + Quellen im Terminal.

---

## 8. Evaluation

`eval/evaluate.py` misst die **Hit@k**-Metrik:
> Für jede Testfrage: Ist unter den `TOP_K` gefundenen Chunks mindestens einer
> aus dem **erwarteten Dokument**? Wenn ja → „Hit".

Das prüft den **Retrieval**-Schritt (findet das System die richtige Quelle?).
Es prüft NICHT, ob das LLM perfekt formuliert – das ist bewusst eine einfache
Metrik für ein Lernprojekt.

**Warum überhaupt messen?** Im Interview willst du sagen können: *„Mein Retrieval
trifft bei den Testfragen zu X % die richtige Quelle"* – statt nur *„es fühlt
sich gut an"*. Messbarkeit ist ein Profi-Signal.

**Profi-Erweiterungen:**
- **Hit@k / Recall@k** über viele Fragen.
- **RAGAS**: misst zusätzlich *Faithfulness* (stützt sich die Antwort wirklich
  auf den Kontext?) und *Answer Relevancy* – mit einem LLM als Bewerter.
- **Re-Ranking**: Treffer mit einem Cross-Encoder nachsortieren.

---

## 9. Interview-Fragen

**„Was ist RAG und warum nutzt man es?"**
> Retrieval-Augmented Generation. Man holt relevante Dokumentstellen und gibt sie
> dem LLM als Kontext. Vorteile: aktuelles/internes Wissen ohne Re-Training,
> Quellenangaben, weniger Halluzinationen, Datenschutz (hier alles lokal).

**„Warum chunken, und wie wählst du die Größe?"**
> Wegen des Embedding-Limits und der Such-Präzision. Größe ist ein Trade-off:
> klein = präzise aber kontextarm, groß = kontextreich aber unpräzise/teurer.
> Ich nutze ~800 Zeichen mit 120 Overlap und teste das per Eval.

**„Was ist ein Embedding / wie funktioniert die Suche?"**
> Ein Vektor, der Bedeutung kodiert. Suche = nächste Nachbarn per Cosine-Distanz.
> Frage und Dokumente müssen mit demselben Modell eingebettet werden.

**„Warum ChromaDB und nicht X?"**
> Lokal, persistent, Metadaten-Support, einfache API – passt zu offline/8 GB.
> Pinecone wäre Cloud (hier verboten), FAISS bräuchte mehr Eigenbau.

**„Wie verhinderst du Halluzinationen?"**
> System-Prompt bindet das Modell an den Kontext + niedrige Temperature +
> Quellenangabe, sodass man Antworten überprüfen kann.

**„Wie würdest du es produktionsreif machen?"**
> Re-Ranking, bessere Embedder, Caching, Streaming, robustere Metriken (RAGAS),
> Authentifizierung der API, Monitoring, inkrementelles Ingest statt Full-Reset.

**„Wofür ist der Overlap?"**
> Damit Inhalte an Chunk-Grenzen nicht verloren gehen und in mindestens einem
> Chunk vollständig vorkommen.

---

## 10. Häufige Fehler

| Symptom | Ursache | Lösung |
|---|---|---|
| `connection refused` bei einer Frage | Ollama-Dienst läuft nicht | Mac-App starten oder `ollama serve` |
| `model "llama3.2:3b" not found` | Modell nicht geladen | `ollama pull llama3.2:3b` |
| Installation schlägt fehl | venv nutzt Python 3.14 | venv mit `python3.12 -m venv .venv` neu anlegen |
| „Keine Dokumente gefunden" | `data/` leer | PDFs/.txt in `data/` legen, dann `python -m src.ingest` |
| Sehr langsam / Speicher voll | 3b zu groß bei vielen Apps | in `config.py` auf `llama3.2:1b`, `TOP_K=3` |
| Antworten ungenau | Chunking/Top_k ungünstig | `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` variieren und `evaluate.py` vergleichen |

---

# 🕸️ Teil 2 — GraphRAG verstehen

Dieser Teil erweitert das obige Vector-RAG um einen **Knowledge Graph**. Alles
hier ist **additiv**: Das Vector-RAG bleibt unangetastet und funktioniert auch,
wenn Neo4j gar nicht läuft.

## 11. Warum überhaupt ein Graph? (Das Kernproblem)

Vector-RAG findet **ähnliche Textstellen**. Das ist stark, hat aber eine
Schwäche: Manche Antworten brauchen Fakten, die **über mehrere Stellen verteilt**
sind und nirgends in EINEM Absatz zusammenstehen.

> **Beispiel:** „Welche Komponenten nutzt Sipwise C5 und wie hängen sie
> zusammen?" Steht auf Seite 5, dass C5 Kamailio nutzt, auf Seite 40, dass
> Kamailio SIP verarbeitet, und auf Seite 80, dass Daten in MySQL liegen, dann
> findet die Vektor-Suche vielleicht nur EINE dieser Stellen. Den **Zusammenhang**
> sieht sie nicht.

Ein **Graph** speichert genau diese Zusammenhänge als Netz:

```
   (Sipwise C5) ──uses──▶ (Kamailio) ──handles──▶ (SIP signaling)
        │
        └──stores data in──▶ (MySQL)
```

Jetzt kann man von „Sipwise C5" aus allen Verbindungen folgen und die Fakten
**einsammeln** – auch wenn sie ursprünglich auf verschiedenen Seiten standen.

**Merksatz:** *Vector Search findet ähnliche Texte, der Graph findet verbundene
Fakten, GraphRAG kombiniert beides.*

## 12. Grundbegriffe Graph (Glossar)

- **Knoten (Node):** Ein „Ding" im Graph (eine Entität), z. B. *Kamailio*. Hat
  bei uns das Label `:Entity` und Eigenschaften (`name`, `display`, `type`).
- **Kante (Relationship/Edge):** Eine Verbindung zwischen zwei Knoten, z. B.
  *uses*. Hat eine Richtung (von Subjekt zu Objekt).
- **Triple:** Ein Fakt aus drei Teilen: **(Subjekt) –[Beziehung]→ (Objekt)**.
  Genau die Bausteine eines Graphen.
- **Entität:** Ein benennbarer Begriff (Komponente, Protokoll, Konzept …).
- **Knowledge Graph:** Das gesamte Netz aus Entitäten und ihren Beziehungen.
- **Cypher:** Die Abfragesprache von Neo4j (das „SQL für Graphen").
- **Traversierung (Traversal):** Das „Entlanggehen" von Kanten von einem Knoten
  zu seinen Nachbarn.
- **Hop:** Ein Schritt entlang einer Kante. „1-Hop" = direkte Nachbarn.

## 13. Warum Neo4j? Warum Cypher?

**Warum Neo4j als Graph-Datenbank?**
| Grund | Erklärung |
|---|---|
| Speziell für Graphen gebaut | Knoten/Kanten sind „first class". Beziehungen zu folgen ist extrem schnell (kein teures JOIN wie in SQL). |
| Lokal & offline per Docker | Bleibt unserem Datenschutz-Prinzip treu. Ein `docker compose up` genügt. |
| Mitgelieferter Browser | http://localhost:7474 zeigt den Graphen **visuell** – perfekt zum Lernen und für die Interview-Demo. |
| Reife, dokumentierte Sprache | Cypher ist lesbar und weit verbreitet. |

**Warum Cypher?** Cypher ist bewusst **visuell** aufgebaut – man „malt" das
Muster, das man sucht, mit ASCII-Pfeilen:

```cypher
//  Finde: ein Knoten s, der über eine REL-Kante r auf einen Knoten o zeigt.
MATCH (s:Entity)-[r:REL]->(o:Entity)
RETURN s.display, r.type, o.display
LIMIT 10
```
- `( )` ist immer ein **Knoten**, `-[ ]->` immer eine **Kante** (mit Richtung).
- `MATCH` = „suche dieses Muster". `RETURN` = „gib das zurück".
Man sieht der Abfrage förmlich an, welche Form sie im Graph sucht. Das ist der
große Lern-Vorteil von Cypher gegenüber abstraktem SQL.

**Die wichtigsten Cypher-Bausteine in diesem Projekt** (alle im Code kommentiert):
- `MERGE` – „finde ODER erstelle" (verhindert Duplikate beim Befüllen).
- `MATCH ... WHERE` – Muster suchen + filtern.
- `DETACH DELETE` – Knoten samt Kanten löschen (für `reset_graph`).
- `CREATE CONSTRAINT ... IS UNIQUE` – Eindeutigkeit + automatischer Index.
- `CONTAINS`, `UNWIND`, `count()`, `startNode()/endNode()` – fürs Retrieval.

## 14. Die GraphRAG-Pipeline Schritt für Schritt

### Bauen (einmalig, `python -m src.graph_ingest`)

**[G1+G2] Laden & Chunking** — *dieselben* Module wie beim Vector-RAG
(`document_loader.py`, `chunker.py`). Kein doppelter Code – beide Welten bauen
auf demselben Text auf.

**[G3] Extraktion** — `graph_extractor.py`
> Jeder Chunk geht durchs **lokale LLM** mit der Bitte: „Zieh die Fakten als
> JSON-Triples heraus." Ein **Beispiel im Prompt** (Few-Shot) sorgt dafür, dass
> das kleine Modell zuverlässig sauberes JSON liefert. `temperature=0` →
> faktentreu, nicht kreativ. Eine **Selbstheilung** schneidet bei kaputtem JSON
> den Bereich von `[` bis `]` heraus, statt aufzugeben.

**[G4] Speichern** — `graph_store.py`
> `write_triples()` schreibt jedes Triple mit `MERGE` nach Neo4j. Unvollständige
> Triples (fehlt Subjekt/Objekt/Beziehung) werden **verworfen** – das LLM liefert
> nicht immer perfekt, und Müll im Graph wollen wir nicht. Namen werden
> **normalisiert** (klein, getrimmt), damit „Kamailio" und „kamailio" derselbe
> Knoten werden.

### Abfragen (bei jeder Frage)

**[G5] Graph-Retrieval** — `graph_retriever.py` (zwei Schritte)
> 1. **Saat-Knoten finden:** Stichwörter aus der Frage ziehen (ohne Stoppwörter)
>    und Entitäten suchen, deren Name dazu passt (`CONTAINS`). Das sind die
>    Einstiegspunkte ins Netz.
> 2. **Beziehungen folgen (1-Hop):** Von den Saat-Knoten aus die direkt
>    verbundenen Fakten einsammeln. Bewusst nur **direkte Nachbarn** – mehr Hops
>    = exponentiell mehr Fakten = das kleine LLM würde überflutet.

**Fusion + Generation** — `hybrid_retriever.py` + `graph_pipeline.py`
> Der Hybrid-Retriever holt **Vektor-Chunks UND Graph-Fakten** und baut einen
> Prompt mit **zwei klar getrennten Abschnitten**:
> ```
> === FAKTEN AUS DEM WISSENSGRAPH ===   ← das "Skelett" der Zusammenhänge
> - Kamailio handles SIP signaling.
> - Sipwise C5 uses Kamailio.
> === PASSENDE TEXTSTELLEN ===          ← das "Fleisch" der Details
> [Textstelle 1 | Quelle: handbook.pdf] ...
> ```
> Das LLM bekommt beide Blickwinkel und formuliert die Antwort.

## 15. Die Fusions-Strategie („Context Fusion") – und warum so

Wir mischen die beiden Quellen **nicht** zu einem gemeinsamen Zahlen-Score,
sondern legen sie als **zwei getrennte Abschnitte** in den Prompt. Begründung:

- **Einfach & transparent:** Keine fragile Umrechnung zwischen zwei völlig
  verschiedenen Maßen (Cosine-Distanz vs. Graph-Trefferzahl). Man sieht jederzeit,
  welcher Teil woher kam.
- **Komplementär:** Graph = Struktur/Zusammenhänge, Text = Details/Formulierungen.
- **Robust (wichtig!):** Findet der Graph nichts (unbekannte Begriffe, Neo4j aus),
  bleibt der Vektor-Teil voll funktionsfähig. GraphRAG wird also **nie schlechter**
  als reines Vector-RAG – im schlimmsten Fall gleich gut.

> Profi-Alternativen fürs Interview: **Reciprocal Rank Fusion** (echtes Verschmelzen
> von Ranglisten) oder ein **Re-Ranker** als zweite Stufe.

## 16. Die Trade-offs (das zentrale Interview-Thema: RAM & Quantisierung)

Auf einem **8-GB-Mac**, auf dem LLM und Graph-DB **gleichzeitig** laufen, ist RAM
das knappste Gut. Bewusst getroffene Kompromisse:

| Trade-off | Entscheidung | Warum |
|---|---|---|
| Container-Runtime | **Colima** statt Docker Desktop | Docker Desktops VM frisst spürbar RAM; Colima ist schlanker. |
| VM-Größe | 2 CPU / **4 GB** | Genug für Neo4j, lässt ~4 GB für natives Ollama + macOS. Ollama läuft NICHT in der VM. |
| Neo4j-Speicher | **~1 GB** (`heap 512m` + `pagecache 512m`) | Unser Doku-Graph ist klein; mehr wäre verschwendetes RAM. |
| Extraktions-Modell | `EXTRACTION_MODEL_NAME` (3b ↔ 1b) | **Kern-Quantisierungs-Abwägung:** kleineres Modell = schneller/weniger RAM, aber gröbere Triples. |
| Chunk-Anzahl | `MAX_CHUNKS_FOR_GRAPH` (Default 150) | Extraktion = 1 LLM-Aufruf/Chunk. Begrenzen → Minuten statt Stunden. |
| Kantenmodell | EIN Typ `:REL` + `type`-Eigenschaft | Dynamische Kantentypen bräuchten APOC → mehr RAM/Komplexität. |
| Traversierung | nur **1 Hop** | Tiefe Pfade = Prompt-Explosion; direkte Nachbarn reichen für Doku-Fragen. |

**Die Quantisierungs-Geschichte fürs Interview:** „Die Entitäts-Extraktion ruft
das LLM einmal pro Chunk auf – das dominiert die Laufzeit. Auf 8 GB RAM ist das
eine klassische Abwägung zwischen Modellgröße/Quantisierung und Qualität: Ein
kleineres, stärker quantisiertes Modell (z. B. 1b) extrahiert deutlich schneller
und mit weniger Speicher, liefert aber gröbere, manchmal unvollständige Triples.
Deshalb ist das Modell eine zentrale, leicht umstellbare Stellschraube in
`config.py`, und unsichere Triples werden beim Schreiben herausgefiltert."

## 17. GraphRAG: typische Interview-Fragen

**„Was ist der Unterschied zwischen Vector-RAG und GraphRAG?"**
> Vector-RAG sucht semantisch ähnliche Textstellen. GraphRAG sucht zusätzlich
> verbundene Fakten in einem Knowledge Graph. Vector ist stark bei „worum geht
> es", Graph bei „wie hängt das zusammen". Mein System kombiniert beide.

**„Wie baust du den Graphen aus unstrukturiertem Text?"**
> Chunks durchs LLM schicken, das Fakten als Triples (Subjekt-Beziehung-Objekt)
> im JSON-Format extrahiert; per `MERGE` dedupliziert nach Neo4j schreiben.

**„Warum Neo4j und nicht eine relationale DB?"**
> Beziehungen zu folgen ist in einem Graph nativ und schnell (keine teuren JOINs),
> und Cypher macht die Muster sichtbar. Plus visueller Browser zum Lernen.

**„Wie kombinierst du die beiden Retriever?"**
> Context Fusion: zwei getrennte Kontext-Abschnitte (Graph-Fakten + Textstellen)
> im Prompt. Transparent, robust, und nie schlechter als reines Vector-RAG.

**„Was war die größte Herausforderung?"**
> RAM auf 8 GB. Lösung: schlanke Runtime (Colima), gedeckeltes Neo4j, kleineres
> Extraktions-Modell als Option, begrenzte Chunk-Anzahl – und das alles in
> `config.py` als bewusste, dokumentierte Stellschrauben.

**„Wie misst du, ob GraphRAG hilft?"**
> `eval/compare_eval.py` misst pro Ansatz die Coverage erwarteter Entitäten im
> Kontext. In meinem Lauf (kleiner ~150-Chunk-Graph) lagen beide bei **73 %** –
> gleichauf in der reinen Coverage. Der Unterschied ist hier **qualitativ**:
> GraphRAG bringt zusätzlich verbundene Fakten in den Prompt. Ehrlich: Ein
> Coverage-Vorsprung bräuchte einen größeren Graphen und gezielt auf
> Mehr-Hop-Zusammenhänge zugeschnittene Fragen. Diese Ehrlichkeit beim Messen
> ist mir wichtiger als ein geschöntes Ergebnis.

---

*Ende. Wenn du eine Stelle vertiefen willst, gib dieses Dokument einem
Assistenten und frag z. B.: „Erklär mir Abschnitt 4 mit einem Zahlenbeispiel"
oder „Geh mit mir `graph_retriever.py` Zeile für Zeile durch".*
