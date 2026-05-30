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
