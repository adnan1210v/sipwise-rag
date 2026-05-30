"""
GraphRAG-HERZSTÜCK: HYBRIDES Retrieval (Vektor-Suche + Graph-Suche kombiniert).

Hier kommt alles zusammen. Wir holen ZWEI Arten von Kontext und führen sie
zusammen, bevor das LLM antwortet:

  A) VEKTOR-Kontext (aus retriever.py / ChromaDB):
     -> die ähnlichsten TEXT-CHUNKS. Stark bei "worum geht es ungefähr?",
        liefert ganze, zusammenhängende Textstellen mit Formulierungen.

  B) GRAPH-Kontext (aus graph_retriever.py / Neo4j):
     -> verbundene FAKTEN rund um die Begriffe der Frage. Stark bei "wie hängt X
        mit Y zusammen?", liefert präzise Beziehungen über mehrere Sätze/Seiten
        hinweg, die in keinem EINZELNEN Chunk zusammenstehen.

------------------------------------------------------------------------------
DIE ZUSAMMENFÜHRUNGS-STRATEGIE (wichtig fürs Interview!)
------------------------------------------------------------------------------
Wir nutzen einen einfachen, gut erklärbaren Ansatz namens "Context Fusion":
Wir mischen die Ergebnisse NICHT zu einem gemeinsamen Score, sondern bauen einen
Prompt mit ZWEI klar getrennten Abschnitten ("Fakten aus dem Wissensgraph" und
"Passende Textstellen"). Das LLM bekommt also beide Blickwinkel und entscheidet
selbst, was es nutzt.

WARUM diese Strategie?
  - Einfach & transparent: keine fragile Score-Normalisierung zwischen zwei ganz
    unterschiedlichen Maßen (Cosine-Distanz vs. Graph-Treffer). Man kann jederzeit
    sehen, welcher Teil aus welcher Quelle kam.
  - Komplementär: Graph-Fakten liefern das "Skelett" der Zusammenhänge, die
    Text-Chunks liefern das "Fleisch" (Details/Formulierungen).
  - Robust: Findet der Graph nichts (z. B. unbekannte Begriffe), bleibt der
    Vektor-Teil voll funktionsfähig – das System verschlechtert sich also nie
    UNTER reines Vektor-RAG, im schlechtesten Fall ist es gleich gut.

(Profi-Alternativen, die man im Interview nennen kann: "Reciprocal Rank Fusion"
zum echten Verschmelzen von Ranglisten, oder ein Re-Ranker als zweite Stufe.)
"""

from . import config
from .retriever import retrieve as vector_retrieve
from .graph_retriever import retrieve_graph_context, facts_to_text


def hybrid_retrieve(question: str, top_k: int | None = None) -> dict:
    """
    Holt Vektor- UND Graph-Kontext zu einer Frage und gibt beides strukturiert
    zurück (noch ohne LLM – das macht der Generator/die Pipeline).

    Rückgabe:
        {
          "chunks": [...],   # Vektor-Treffer (wie im reinen Vector-RAG)
          "graph":  { "seeds": [...], "facts": [...] },
          "fused_context": "....",  # fertig formatierter Text für den LLM-Prompt
        }
    """
    # --- Teil A: Vektor-Suche (unverändert das bestehende Retrieval) ---------
    chunks = vector_retrieve(question, top_k=top_k)

    # --- Teil B: Graph-Suche -------------------------------------------------
    # Schlägt der Graph fehl (z. B. Neo4j aus), soll NICHT das ganze System
    # crashen – dann liefern wir einfach leeren Graph-Kontext (= reines Vector-RAG).
    try:
        graph = retrieve_graph_context(question)
    except Exception as e:
        print(f"  ⚠️  Graph-Retrieval übersprungen (Neo4j-Problem?): {e}")
        graph = {"seeds": [], "facts": []}

    # --- Zusammenführen zu EINEM Kontext-Text für das LLM --------------------
    fused_context = _build_fused_context(chunks, graph)

    return {"chunks": chunks, "graph": graph, "fused_context": fused_context}


def _build_fused_context(chunks: list[dict], graph: dict) -> str:
    """
    Baut den kombinierten Kontext-Text mit zwei klar getrennten Abschnitten.

    Reihenfolge bewusst: zuerst die GRAPH-FAKTEN (kompaktes "Skelett" der
    Zusammenhänge), dann die TEXT-CHUNKS (ausführliche Belege). So sieht das
    Modell zuerst die Struktur und dann die Details.
    """
    parts = []

    facts_text = facts_to_text(graph.get("facts", []))
    if facts_text:
        parts.append(
            "=== FAKTEN AUS DEM WISSENSGRAPH (verbundene Beziehungen) ===\n"
            + facts_text
        )

    if chunks:
        chunk_blocks = []
        for i, c in enumerate(chunks, start=1):
            chunk_blocks.append(
                f"[Textstelle {i} | Quelle: {c['source']}]\n{c['text']}"
            )
        parts.append(
            "=== PASSENDE TEXTSTELLEN (ähnliche Dokumentabschnitte) ===\n"
            + "\n\n".join(chunk_blocks)
        )

    # Doppelte Leerzeile trennt die beiden großen Blöcke optisch im Prompt.
    return "\n\n".join(parts)


# -----------------------------------------------------------------------------
# Schnelltest:  python -m src.hybrid_retriever "How does Kamailio handle SIP?"
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    frage = " ".join(sys.argv[1:]) or "What is Kamailio?"
    print(f"❓ Frage: {frage}\n")

    result = hybrid_retrieve(frage)

    print(f"📄 Vektor-Treffer: {len(result['chunks'])} Chunks")
    print(f"🌱 Graph-Saat-Knoten: {len(result['graph']['seeds'])}")
    print(f"🔗 Graph-Fakten: {len(result['graph']['facts'])}\n")
    print("=== ZUSAMMENGEFÜHRTER KONTEXT (geht so ans LLM) ===\n")
    print(result["fused_context"] or "(leer)")

    from .graph_store import close_driver
    close_driver()
