"""
Schritt 3 der Pipeline: EMBEDDINGS erzeugen.

Ein "Embedding" ist ein Zahlenvektor (hier 384 Zahlen), der die BEDEUTUNG eines
Textes repräsentiert. Texte mit ähnlicher Bedeutung liegen im Vektorraum nah
beieinander. Dadurch können wir später "nach Sinn" suchen statt nur nach exakten
Wörtern (z. B. findet "Wie konfiguriere ich SIP?" auch Text über "peering setup").

Designentscheidung: Wir laden das Modell EINMAL und halten es im Speicher
(Singleton-Muster). Das Laden dauert ein paar Sekunden – das wollen wir nicht
bei jeder Anfrage neu machen.
"""

from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME

# Modul-globale Variable: das geladene Modell wird hier zwischengespeichert.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lädt das Embedding-Modell beim ersten Aufruf, danach aus dem Cache."""
    global _model
    if _model is None:
        # Beim allerersten Mal lädt sentence-transformers das Modell aus dem
        # Internet herunter und legt es lokal ab (~/.cache). Danach: offline.
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Wandelt eine Liste von Texten (z. B. alle Chunks) in Vektoren um.

    Wird bei der Befüllung der DB benutzt. encode() verarbeitet die Texte in
    Batches – effizient und speicherschonend, auch bei vielen Chunks.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,           # Fortschrittsbalken beim Einspeisen
        convert_to_numpy=True,
    )
    # ChromaDB erwartet einfache Python-Listen, kein NumPy-Array.
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Wandelt EINE Nutzerfrage in einen Vektor um (für die Suche, Schritt 5).

    Wichtig: Die Frage muss mit DEMSELBEN Modell eingebettet werden wie die
    Dokumente – sonst liegen Frage und Antwort in unterschiedlichen "Sprachen"
    des Vektorraums und die Suche funktioniert nicht.
    """
    model = get_model()
    return model.encode(query, convert_to_numpy=True).tolist()
