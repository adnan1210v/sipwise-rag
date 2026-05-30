"""
Schritt 2 der Pipeline: Text in CHUNKS zerlegen.

WARUM überhaupt chunken?
1. Embedding-Modelle haben ein Eingabe-Limit (MiniLM ~256 Wörter). Ein ganzes
   Handbuch passt nicht in einen Vektor.
2. Retrieval wird präziser: Wir wollen GENAU den Absatz finden, der die Frage
   beantwortet – nicht ein 300-Seiten-Dokument.
3. Das LLM bekommt nur die wenigen relevanten Häppchen statt des ganzen Buchs
   (spart RAM und Zeit, und lenkt das Modell weniger ab).

Strategie hier: einfacher, gut nachvollziehbarer Splitter mit fester Größe und
Overlap. Wir versuchen, an Absatz-/Satzgrenzen zu trennen, damit Chunks nicht
mitten im Wort enden. (Industrieüblich gibt es ausgefeiltere Splitter, z. B. in
LangChain – wir bauen es bewusst selbst, damit du JEDE Zeile verstehst.)
"""

from .config import CHUNK_SIZE, CHUNK_OVERLAP


def _clean(text: str) -> str:
    """Normalisiert Whitespace, damit Chunks sauber sind."""
    # Viele aufeinanderfolgende Leerzeilen/Spaces zu einem Leerzeichen verdichten
    # würde Absatzstruktur zerstören – wir entfernen nur überflüssige Spaces
    # pro Zeile und trimmen Ränder.
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Zerlegt EINEN Dokumenttext in überlappende Chunks.

    Rückgabe: Liste von Dicts:
        {"text": "...", "source": "handbook.pdf", "chunk_index": 0}
    Den chunk_index und die Quelle behalten wir als "Metadaten" – damit kann
    ChromaDB später jede Antwort auf eine konkrete Stelle zurückführen.
    """
    text = _clean(text)
    chunks: list[dict] = []

    start = 0
    chunk_index = 0
    text_length = len(text)

    while start < text_length:
        # Vorläufiges Ende des aktuellen Chunks.
        end = start + CHUNK_SIZE

        # Wenn wir nicht schon am Dokumentende sind, versuchen wir an einer
        # "natürlichen" Grenze zu schneiden (Absatz > Satzende > Leerzeichen),
        # damit kein Wort/Satz mittendrin abgehackt wird.
        if end < text_length:
            window = text[start:end]
            # Wir suchen die letzte gute Trennstelle im Fenster.
            for separator in ["\n\n", "\n", ". ", " "]:
                cut = window.rfind(separator)
                # cut > CHUNK_SIZE * 0.5 -> nur trennen, wenn die Grenze nicht
                # ganz am Anfang liegt (sonst würden Chunks winzig).
                if cut != -1 and cut > CHUNK_SIZE * 0.5:
                    end = start + cut + len(separator)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        # Nächster Start: um (Chunkgröße - Overlap) weiterrücken.
        # Der Overlap sorgt dafür, dass Inhalt an Chunk-Grenzen nicht verloren
        # geht. max(..., start+1) verhindert eine Endlosschleife.
        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Wendet chunk_text auf eine ganze Liste geladener Dokumente an."""
    all_chunks: list[dict] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc["text"], doc["source"]))
    return all_chunks
