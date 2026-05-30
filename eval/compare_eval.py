"""
VERGLEICHENDE EVALUATION: Vector-RAG vs. GraphRAG – messbar gemacht.

Das bestehende eval/evaluate.py misst NUR das Vektor-Retrieval (Hit@k auf
Dateinamen). Dieses Skript ergänzt das um eine GRAPH-orientierte Messung und
stellt beide Ansätze gegenüber – an Fragen, die MEHRERE VERBUNDENE FAKTEN
brauchen (genau dort soll GraphRAG glänzen).

WAS messen wir?
  Für jede Testfrage haben wir eine Liste ERWARTETER ENTITÄTEN (Begriffe, die in
  einer guten Antwort vorkommen sollten, z. B. "kamailio", "mysql"). Wir prüfen
  pro Ansatz, WIE VIELE davon der jeweilige Kontext bzw. die Antwort abdeckt:

    - Vector-RAG:  Kommen die erwarteten Entitäten in den abgerufenen CHUNKS vor?
    - GraphRAG:    Kommen sie in den GRAPH-FAKTEN + Chunks vor?

  Daraus bilden wir eine einfache "Coverage" (Abdeckung) = gefundene / erwartete.
  Höhere Coverage = der Ansatz trägt mehr der nötigen verbundenen Fakten zusammen.

WARUM diese Metrik (und ihre Grenzen)?
  Sie ist bewusst SIMPEL und gut erklärbar (Lernprojekt). Sie misst, ob die
  richtigen "Bausteine" im Kontext landen – nicht, ob das LLM perfekt formuliert.
  Profi-Erweiterung: LLM-as-a-judge / RAGAS (Faithfulness, Answer-Relevancy).

Aufruf (Neo4j + ChromaDB müssen befüllt sein):
    python -m eval.compare_eval
"""

import json
from pathlib import Path

from src.retriever import retrieve
from src.hybrid_retriever import hybrid_retrieve
from src.graph_store import close_driver

TEST_FILE = Path(__file__).resolve().parent / "graph_test_questions.json"


def _coverage(found_text: str, expected: list[str]) -> tuple[int, int]:
    """
    Zählt, wie viele erwartete Entitäten als Teilstring im Text vorkommen.

    Rückgabe: (gefunden, erwartet_gesamt). Alles klein verglichen -> robust.
    """
    text = found_text.lower()
    found = sum(1 for e in expected if e.lower() in text)
    return found, len(expected)


def run_comparison():
    test_cases = json.loads(TEST_FILE.read_text(encoding="utf-8"))

    print("=" * 72)
    print("VERGLEICH:  Vector-RAG  vs.  GraphRAG   (Coverage erwarteter Entitäten)")
    print("=" * 72)

    vec_found_total = vec_exp_total = 0
    graph_found_total = graph_exp_total = 0

    for case in test_cases:
        question = case["question"]
        expected = case["expected_entities"]

        # --- Vector-RAG: erwartete Entitäten in den abgerufenen Chunks? ------
        chunks = retrieve(question)
        vec_text = " ".join(c["text"] for c in chunks)
        vf, ve = _coverage(vec_text, expected)
        vec_found_total += vf
        vec_exp_total += ve

        # --- GraphRAG: in Graph-Fakten + Chunks? ----------------------------
        hybrid = hybrid_retrieve(question)
        graph_text = hybrid["fused_context"]
        gf, ge = _coverage(graph_text, expected)
        graph_found_total += gf
        graph_exp_total += ge

        print(f"\n❓ {question}")
        print(f"   erwartete Entitäten: {expected}")
        print(f"   Vector-RAG : {vf}/{ve} abgedeckt")
        print(f"   GraphRAG   : {gf}/{ge} abgedeckt  "
              f"(Graph-Fakten: {len(hybrid['graph']['facts'])})")

    # --- Gesamtergebnis -----------------------------------------------------
    vec_score = vec_found_total / vec_exp_total if vec_exp_total else 0
    graph_score = graph_found_total / graph_exp_total if graph_exp_total else 0

    print("\n" + "=" * 72)
    print(f"GESAMT  Vector-RAG : {vec_found_total}/{vec_exp_total}  "
          f"({vec_score:.0%} Coverage)")
    print(f"GESAMT  GraphRAG   : {graph_found_total}/{graph_exp_total}  "
          f"({graph_score:.0%} Coverage)")
    print("=" * 72)
    print("\nLesart: Höhere Coverage = mehr der nötigen verbundenen Fakten landen "
          "im\nKontext. Bei reinen Faktenfragen liegen beide oft gleichauf; bei "
          "Fragen\nüber ZUSAMMENHÄNGE sollte GraphRAG vorne liegen.")

    close_driver()


if __name__ == "__main__":
    run_comparison()
