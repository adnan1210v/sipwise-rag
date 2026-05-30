"""
VERGLEICHS-TOOL: dieselbe Frage einmal mit reinem Vector-RAG und einmal mit
GraphRAG beantworten – direkt nebeneinander.

Das ist dein DEMO-Werkzeug fürs Interview: Es macht den Unterschied der beiden
Ansätze sichtbar, statt ihn nur zu behaupten.

Aufruf:
    python -m src.compare "How does Kamailio interact with the database?"

Tipp für eine überzeugende Demo: Wähle eine Frage, die MEHRERE verbundene Fakten
braucht (z. B. "Wie hängen Komponente A, B und C zusammen?"). Genau da spielt der
Graph seine Stärke aus, weil solche Zusammenhänge selten in EINEM Textabschnitt
stehen.
"""

import sys

from .pipeline import answer_question            # reines Vector-RAG
from .graph_pipeline import answer_question_graph  # GraphRAG (hybrid)
from .graph_store import close_driver


def compare(question: str):
    print("=" * 70)
    print(f"❓ Frage: {question}")
    print("=" * 70)

    # --- 1) Reines Vector-RAG ------------------------------------------------
    print("\n" + "─" * 70)
    print("①  NUR VECTOR-RAG  (sucht ähnliche Textstellen)")
    print("─" * 70)
    vec = answer_question(question)
    print(vec["answer"])
    print("\n   Quellen:", ", ".join(
        f"{s['source']}#{s['chunk_index']}" for s in vec["sources"]
    ) or "(keine)")

    # --- 2) GraphRAG (hybrid) ------------------------------------------------
    print("\n" + "─" * 70)
    print("②  GRAPHRAG  (Vektor + verbundene Fakten aus dem Wissensgraph)")
    print("─" * 70)
    g = answer_question_graph(question)
    print(g["answer"])
    print("\n   Einstiegs-Entitäten:", ", ".join(g["seed_entities"]) or "(keine)")
    print(f"   Genutzte Graph-Fakten: {len(g['graph_facts'])}")
    if g["graph_facts"]:
        # Die ersten paar Fakten zeigen – macht greifbar, was der Graph beisteuert.
        print("   Beispiel-Fakten:")
        for f in g["graph_facts"][:5]:
            pred = (f.get("predicate") or "").replace("_", " ")
            print(f"     • {f['subject']} {pred} {f['object']}")

    # --- Hinweis zur Interpretation -----------------------------------------
    print("\n" + "=" * 70)
    print("💡 Worauf achten? GraphRAG kann Zusammenhänge benennen, die über "
          "mehrere\n   Textstellen verteilt sind – das reine Vector-RAG sieht "
          "nur einzelne\n   ähnliche Abschnitte. Bei einfachen Faktenfragen sind "
          "beide oft ähnlich gut.")
    print("=" * 70)


if __name__ == "__main__":
    frage = " ".join(sys.argv[1:])
    if not frage:
        print('Benutzung: python -m src.compare "Deine Frage hier"')
        sys.exit(1)
    try:
        compare(frage)
    finally:
        # Egal was passiert: Neo4j-Verbindung sauber schließen.
        close_driver()
