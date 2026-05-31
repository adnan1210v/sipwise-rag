"""
GraphRAG-PIPELINE: beantwortet eine Frage mit HYBRIDEM Retrieval (Vektor+Graph).

Das ist das Gegenstück zu pipeline.py (reines Vector-RAG). Beide haben dieselbe
"Form": Frage rein -> {answer, sources, ...} raus. Dadurch kann das Vergleichs-
und das Eval-Skript beide gleich behandeln.

Ablauf:
  1. hybrid_retrieve()  -> Vektor-Chunks + Graph-Fakten, fertig fusioniert.
  2. LLM antwortet NUR auf Basis dieses kombinierten Kontexts.
"""

import ollama

from . import config
from .generator import detect_answer_language, polish_known_model_artifacts
from .hybrid_retriever import hybrid_retrieve


# Eigener System-Prompt für GraphRAG: erklärt dem Modell, dass es ZWEI Arten von
# Kontext bekommt (Graph-Fakten + Textstellen) und beide nutzen darf. Ansonsten
# dieselbe Anti-Halluzinations-Haltung wie im Vector-RAG.
GRAPH_SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Assistent für die technische Dokumentation von "
    "Sipwise C5.\n"
    "Du bekommst zwei Arten von Kontext:\n"
    "  (1) FAKTEN AUS DEM WISSENSGRAPH – kurze, verbundene Beziehungen.\n"
    "  (2) PASSENDE TEXTSTELLEN – längere Originalabschnitte aus der Doku.\n"
    "Regeln:\n"
    "1. Stütze deine Fakten ausschließlich auf den bereitgestellten Kontext "
    "(Graph-Fakten UND Textstellen). Erfinde nichts.\n"
    "2. Nutze die Graph-Fakten, um Zusammenhänge zwischen Komponenten zu "
    "erklären, und die Textstellen für Details.\n"
    "3. Du darfst vereinfachen und für Einsteiger erklären, solange es faktisch "
    "korrekt bleibt.\n"
    "4. Antworte exakt in der Antwortsprache, die im Prompt angegeben ist.\n"
    "5. Steht die Antwort nicht im Kontext, sage das ehrlich."
)


def answer_question_graph(question: str, top_k: int | None = None) -> dict:
    """
    Beantwortet eine Frage über die GraphRAG-Pipeline (hybrid).

    Rückgabe (bewusst ähnlich zu pipeline.answer_question, plus Graph-Infos):
        {
          "question": "...",
          "answer": "...",
          "sources": [ {source, chunk_index, distance}, ... ],  # Vektor-Quellen
          "graph_facts": [ {subject, predicate, object}, ... ],  # genutzte Fakten
          "seed_entities": [ "kamailio", ... ],                  # Einstiegspunkte
        }
    """
    retrieval = hybrid_retrieve(question, top_k=top_k)
    chunks = retrieval["chunks"]
    graph = retrieval["graph"]
    fused_context = retrieval["fused_context"]

    # Wenn WEDER Chunks NOCH Graph-Fakten da sind, gar nicht erst das LLM bemühen.
    if not fused_context.strip():
        return {
            "question": question,
            "answer": "Es ist kein Kontext vorhanden. Bitte zuerst die "
                      "Wissensbasis befüllen (python -m src.ingest und "
                      "python -m src.graph_ingest).",
            "sources": [],
            "graph_facts": [],
            "seed_entities": [],
        }

    # Prompt bauen: kombinierter Kontext + Frage.
    answer_language = detect_answer_language(question)
    prompt = (
        f"{fused_context}\n\n"
        f"Frage: {question}\n"
        f"Antwortsprache: {answer_language}\n\n"
        f"Antwort (nur basierend auf dem Kontext oben):"
    )

    client = ollama.Client(
        host=config.OLLAMA_HOST,
        timeout=config.OLLAMA_TIMEOUT_SECONDS,
    )
    response = client.chat(
        model=config.LLM_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": GRAPH_SYSTEM_PROMPT
                + f"\n\nAktuelle Antwortsprache: {answer_language}. "
                  f"Du MUSST die Antwort in {answer_language} schreiben.",
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.1},
    )
    answer = polish_known_model_artifacts(response["message"]["content"])

    # Quellen der Vektor-Treffer (wie im reinen Vector-RAG) zusammenstellen.
    sources = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "distance": round(c["distance"], 4),
        }
        for c in chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "graph_facts": graph.get("facts", []),
        "seed_entities": [s["display"] for s in graph.get("seeds", [])],
    }


# -----------------------------------------------------------------------------
# Schnelltest:  python -m src.graph_pipeline "How does Kamailio handle SIP?"
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    frage = " ".join(sys.argv[1:]) or "What is Kamailio?"
    print(f"❓ Frage: {frage}\n… denke nach (GraphRAG: Vektor + Graph) …\n")

    result = answer_question_graph(frage)

    print("💬 Antwort:")
    print(result["answer"])

    print(f"\n🌱 Einstiegs-Entitäten im Graph: "
          f"{', '.join(result['seed_entities']) or '(keine)'}")
    print(f"🔗 Genutzte Graph-Fakten: {len(result['graph_facts'])}")
    print("\n📚 Text-Quellen:")
    for s in result["sources"]:
        print(f"   • {s['source']}  (Chunk {s['chunk_index']}, "
              f"Distanz {s['distance']})")

    from .graph_store import close_driver
    close_driver()
