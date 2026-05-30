"""
Zentrale Konfiguration des RAG-Systems.

Idee: ALLE "Stellschrauben" liegen an einem Ort. Wenn du etwas ausprobieren
willst (anderes Modell, größere Chunks, mehr Treffer), änderst du es hier –
nicht verstreut im Code. Das macht Experimente nachvollziehbar, was bei
RAG entscheidend ist (kleine Parameteränderungen = große Qualitätsunterschiede).
"""

from pathlib import Path

# --- Pfade -------------------------------------------------------------------
# Wir leiten alle Pfade vom Projekt-Hauptordner ab, damit das Projekt von
# überall aus startbar ist (egal aus welchem Verzeichnis du es aufrufst).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"          # hier legst DU deine PDFs/Texte ab
CHROMA_DIR = PROJECT_ROOT / "chroma_db"   # hier speichert ChromaDB auf Platte

# --- Embedding-Modell (Schritt 3) -------------------------------------------
# all-MiniLM-L6-v2: ~90 MB, erzeugt 384-dimensionale Vektoren.
# WARUM dieses Modell? Es ist winzig (passt locker in 8 GB RAM), sehr schnell
# und für englische technische Texte (wie das Sipwise-Handbuch) gut genug.
# Es läuft komplett offline, nachdem es einmal heruntergeladen wurde.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- LLM (Schritt 6, via Ollama) --------------------------------------------
# qwen2.5:1.5b (~986 MB) war auf diesem Mac bereits installiert und passt
# bestens in 8 GB RAM. Ollama lädt das Modell in einem EIGENEN Prozess und
# nur bei Bedarf in den Speicher.
# Bessere (aber größere) Alternative, falls du sie laden willst:
#   ollama pull llama3.2:3b   ->  dann hier "llama3.2:3b" eintragen.
# Das ist der einzige nötige Code-Eingriff, um das LLM zu wechseln.
LLM_MODEL_NAME = "llama3.2:3b"

# --- Chunking (Schritt 2) ----------------------------------------------------
# CHUNK_SIZE = Länge eines Textstücks in ZEICHEN (nicht Tokens, der Einfachheit
# halber). ~800 Zeichen ≈ 1–2 Absätze. WARUM nicht größer? Kleinere Chunks =
# präzisere Treffer beim Retrieval; ABER zu klein = Kontext geht verloren.
# 800 ist ein bewährter Kompromiss für technische Doku.
CHUNK_SIZE = 800
# CHUNK_OVERLAP = wie viele Zeichen sich zwei aufeinanderfolgende Chunks teilen.
# WARUM Overlap? Damit ein Satz, der genau an einer Chunk-Grenze liegt, nicht
# zerrissen wird und in mindestens einem Chunk vollständig vorkommt.
CHUNK_OVERLAP = 120

# --- Retrieval (Schritt 5) ---------------------------------------------------
# TOP_K = wie viele der ähnlichsten Chunks wir als Kontext an das LLM geben.
# Mehr Kontext = potenziell bessere Antwort, aber mehr RAM/Zeit und Gefahr,
# dass irrelevanter Text das LLM ablenkt. 4 ist ein guter Startwert.
TOP_K = 4

# Name der Collection (Tabelle) in ChromaDB.
COLLECTION_NAME = "sipwise_docs"


# =============================================================================
# GraphRAG-Erweiterung (Knowledge Graph mit Neo4j)
# =============================================================================
# Diese Einstellungen betreffen NUR den neuen Graph-Teil. Das bestehende
# Vector-RAG oben funktioniert davon völlig unabhängig weiter.
import os  # nur hier nötig: erlaubt, Werte per Umgebungsvariable zu überschreiben

# --- Neo4j-Verbindung -------------------------------------------------------
# "bolt://" ist das schnelle Binärprotokoll von Neo4j (Port 7687). localhost,
# weil die DB als Docker-Container auf DEINEM Rechner läuft – nichts geht ins
# Internet (bleibt unserem Offline-Prinzip treu).
# os.getenv(..., default): falls eine Umgebungsvariable gesetzt ist, gewinnt
# sie – sonst der Default. So kannst du z. B. das Passwort setzen, ohne Code
# zu ändern. Die Defaults passen exakt zur docker-compose.yml.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "sipwise123")

# --- LLM für die Graph-EXTRAKTION (Knoten & Kanten aus Text) ----------------
# Eigene Einstellung, NICHT dasselbe wie LLM_MODEL_NAME oben! Begründung:
# Die Extraktion ruft das LLM EINMAL PRO CHUNK auf (bei hunderten Chunks also
# sehr oft) -> das ist der teuerste, langsamste Schritt von GraphRAG.
# Trade-off (wichtig fürs Interview):
#   - llama3.2:3b  -> genauer, aber langsamer/speicherhungriger (Default).
#   - llama3.2:1b  -> ~2x schneller, weniger RAM, dafür etwas gröbere Triples.
# Wenn die Extraktion zu lange dauert oder RAM knapp wird: hier auf "llama3.2:1b"
# umstellen (vorher: ollama pull llama3.2:1b).
EXTRACTION_MODEL_NAME = os.getenv("EXTRACTION_MODEL_NAME", LLM_MODEL_NAME)

# --- Steuerung der Extraktion ----------------------------------------------
# Aus wie vielen Chunks extrahieren wir? Bei einem 34-MB-Handbuch entstehen
# SEHR viele Chunks. Jeden durchs LLM zu schicken kann auf 8 GB RAM Stunden
# dauern. Für ein Demo-/Lernprojekt begrenzen wir die Anzahl, damit der Graph
# in Minuten statt Stunden steht. None = wirklich ALLE Chunks (Vollausbau).
# Trade-off: mehr Chunks = reichhaltigerer Graph, aber deutlich längere Laufzeit.
MAX_CHUNKS_FOR_GRAPH = int(os.getenv("MAX_CHUNKS_FOR_GRAPH", "150"))

# --- Graph-Retrieval --------------------------------------------------------
# Wie viele "Saat"-Knoten suchen wir pro Frage als Einstiegspunkte in den Graph?
GRAPH_SEED_ENTITIES = 5
# Wie viele verbundene Fakten (Triples) geben wir maximal als Kontext ans LLM?
# Begrenzung, damit der Prompt nicht explodiert und das kleine LLM fokussiert
# bleibt.
GRAPH_MAX_FACTS = 30
