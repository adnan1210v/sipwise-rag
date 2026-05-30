"""
Schritt 1 der Pipeline: Dokumente LADEN.

Aufgabe: Aus dem Ordner data/ alle PDF- und Text-Dateien einlesen und ihren
reinen Text zurückgeben. Pro Dokument merken wir uns den Dateinamen als
"Quelle" (source). Diese Quelle reichen wir durch die ganze Pipeline durch,
damit die spätere Antwort belegen kann, AUS WELCHEM Dokument sie stammt –
ein zentrales Qualitätsmerkmal von RAG (Nachvollziehbarkeit / "grounding").
"""

from pathlib import Path
from pypdf import PdfReader


def _read_pdf(path: Path) -> str:
    """Liest den Text aus einer einzelnen PDF-Datei Seite für Seite aus."""
    reader = PdfReader(str(path))
    # Wir hängen den Text aller Seiten zusammen. extract_text() kann bei
    # manchen Seiten None liefern (z. B. reine Bildseiten) -> mit "" absichern.
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_txt(path: Path) -> str:
    """Liest eine einfache Text-/Markdown-Datei."""
    # encoding="utf-8" und errors="ignore", damit ungewöhnliche Zeichen nicht
    # das ganze Laden abbrechen.
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(data_dir: Path) -> list[dict]:
    """
    Lädt alle unterstützten Dateien aus data_dir.

    Rückgabe: Liste von Dicts der Form
        {"source": "handbook.pdf", "text": "..."}
    'source' = Dateiname (dient später als Quellenangabe).
    """
    documents: list[dict] = []

    # sorted() -> deterministische Reihenfolge, damit Läufe reproduzierbar sind.
    for path in sorted(data_dir.iterdir()):
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            text = _read_pdf(path)
        elif suffix in {".txt", ".md"}:
            text = _read_txt(path)
        else:
            # Unbekannte Dateitypen (Bilder, .DS_Store, ...) überspringen.
            continue

        # Leere oder kaputte Dateien überspringen – sonst erzeugen wir leere
        # Chunks, die nur Rauschen in die Datenbank bringen.
        if text.strip():
            documents.append({"source": path.name, "text": text})

    return documents
