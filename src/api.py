"""
Schritt 7: die FastAPI-SCHNITTSTELLE.

Stellt die RAG-Pipeline als Web-Endpoint bereit. Damit kannst du Fragen per
HTTP stellen (z. B. aus dem Browser, curl, oder einer kleinen Web-UI) –
schöner Beleg fürs Bewerbungsgespräch, dass das System "produktnah" nutzbar ist.

Starten:
    uvicorn src.api:app --reload
Dann im Browser:  http://127.0.0.1:8000/docs   (automatische API-Doku)
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .pipeline import answer_question
from . import config

# Pfad zur einfachen Web-Oberfläche (HTML-Datei im Ordner web/).
WEB_INDEX = config.PROJECT_ROOT / "web" / "index.html"

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
    question: str
    top_k: int | None = None  # optional: Anzahl Kontext-Chunks überschreiben


class Source(BaseModel):
    source: str
    chunk_index: int
    distance: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


# --- Endpoints ---------------------------------------------------------------
@app.get("/")
def home():
    """Liefert die einfache Web-Oberfläche (Textfeld zum Fragen stellen)."""
    return FileResponse(WEB_INDEX)


@app.get("/health")
def health():
    """Einfacher Health-Check: zeigt, welche Modelle konfiguriert sind."""
    return {
        "status": "ok",
        "llm": config.LLM_MODEL_NAME,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
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
    return answer_question(request.question, top_k=request.top_k)
