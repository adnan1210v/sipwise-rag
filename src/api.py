"""
Schritt 7: die FastAPI-SCHNITTSTELLE.

Stellt die RAG-Pipeline als Web-Endpoint bereit. Damit kannst du Fragen per
HTTP stellen (z. B. aus dem Browser, curl, oder einer kleinen Web-UI) –
schöner Beleg fürs Bewerbungsgespräch, dass das System "produktnah" nutzbar ist.

Starten:
    uvicorn src.api:app --reload
Dann im Browser:  http://127.0.0.1:8000/docs   (automatische API-Doku)
"""

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline import answer_question
from .graph_pipeline import answer_question_graph
from .graph_store import graph_stats, fetch_graph, ping as neo4j_ping
from .vector_store import get_collection
from . import config

# Pfade zur einfachen Web-Oberfläche (statische Dateien im Ordner web/).
WEB_DIR = config.PROJECT_ROOT / "web"
WEB_INDEX = WEB_DIR / "index.html"
WEB_GRAPH = WEB_DIR / "graph.html"
WEB_VENDOR = WEB_DIR / "vendor"

# FastAPI-App mit Titel/Beschreibung -> taucht in der /docs-Oberfläche auf.
app = FastAPI(
    title="Sipwise C5 RAG",
    description="Lokales, offline RAG-System über die Sipwise-C5-Dokumentation.",
    version="1.0.0",
)


# --- Datenmodelle (Pydantic) -------------------------------------------------
# Pydantic validiert automatisch die Ein-/Ausgaben. Schickt jemand z. B. keine
# 'question', gibt FastAPI von selbst eine saubere Fehlermeldung zurück.
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    source: str
    chunk_index: int
    distance: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class GraphFact(BaseModel):
    subject: str
    predicate: str
    object: str
    source_doc: str | None = None
    chunk_index: int | None = None


class GraphQueryResponse(QueryResponse):
    graph_facts: list[GraphFact]
    seed_entities: list[str]


# Datenmodelle für die interaktive Graph-Visualisierung (GET /graph_data).
class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    degree: int


class GraphLink(BaseModel):
    source: str
    target: str
    type: str
    source_doc: str | None = None
    chunk_index: int | None = None


class GraphStats(BaseModel):
    nodes: int
    edges: int
    truncated: bool


class GraphDataResponse(BaseModel):
    available: bool
    message: str | None = None
    nodes: list[GraphNode]
    links: list[GraphLink]
    stats: GraphStats


# Die vendorte JS-Bibliothek (force-graph) lokal ausliefern -> bleibt offline,
# kein CDN zur Laufzeit nötig. Mount nur, wenn der Ordner existiert (er enthält
# die committete force-graph.min.js), damit der Import nie an einem fehlenden
# Verzeichnis scheitert.
if WEB_VENDOR.is_dir():
    app.mount("/vendor", StaticFiles(directory=str(WEB_VENDOR)), name="vendor")


# --- Endpoints ---------------------------------------------------------------
@app.get("/")
def home():
    """Liefert die einfache Web-Oberfläche (Textfeld zum Fragen stellen)."""
    return FileResponse(WEB_INDEX)


@app.get("/graph")
def graph_page():
    """Liefert die interaktive Graph-Visualisierung (Obsidian-artiges Netz)."""
    return FileResponse(WEB_GRAPH)


@app.get("/graph_data", response_model=GraphDataResponse)
def graph_data():
    """
    JSON-Datenquelle für die Graph-Ansicht: alle Knoten + Kanten aus Neo4j.

    Gibt IMMER eine formgleiche Antwort zurück (auch wenn Neo4j aus ist oder der
    Graph leer ist) – das Frontend zeigt dann einen freundlichen Hinweis an.
    Die Obergrenzen kommen aus config (GRAPH_VIEW_MAX_NODES/EDGES).
    """
    return fetch_graph()


@app.get("/health")
def health():
    """Health-Check für API, ChromaDB, Ollama und optional Neo4j."""
    vector_count = None
    vector_ok = False
    try:
        vector_count = get_collection().count()
        vector_ok = True
    except Exception:
        pass

    ollama_ok = False
    try:
        client = ollama.Client(
            host=config.OLLAMA_HOST,
            timeout=config.OLLAMA_TIMEOUT_SECONDS,
        )
        client.list()
        ollama_ok = True
    except Exception:
        pass

    neo4j_ok = neo4j_ping()
    stats = None
    if neo4j_ok:
        try:
            stats = graph_stats()
        except Exception:
            stats = None

    return {
        "status": "ok" if vector_ok and ollama_ok else "degraded",
        "llm": config.LLM_MODEL_NAME,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "vector_store": {"ok": vector_ok, "chunks": vector_count},
        "ollama": {"ok": ollama_ok, "host": config.OLLAMA_HOST},
        "neo4j": {"ok": neo4j_ok, "stats": stats},
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Haupt-Endpoint: nimmt eine Frage entgegen und gibt Antwort + Quellen zurück.

    Beispiel (curl):
        curl -X POST http://127.0.0.1:8000/query \\
             -H "Content-Type: application/json" \\
             -d '{"question": "How do I configure SIP peering?"}'
    """
    try:
        return answer_question(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query_graph", response_model=GraphQueryResponse)
def query_graph_endpoint(request: QueryRequest):
    """GraphRAG-Endpoint: Vektor-Kontext plus Neo4j-Fakten."""
    try:
        return answer_question_graph(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
