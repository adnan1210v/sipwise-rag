"""
Einfaches EVALUIERUNGS-Skript: misst die Qualität des RETRIEVAL-Schritts.

WARUM Evaluation? "Funktioniert mein RAG?" ist keine Bauchgefühl-Frage. Im
Bewerbungsgespräch beeindruckt, wenn du Qualität MISST statt nur behauptest.

Was messen wir hier? Die Metrik "Hit@k":
  Für jede Testfrage prüfen wir, ob unter den top_k abgerufenen Chunks
  mindestens einer aus dem erwarteten Dokument stammt (Stichwort-Abgleich auf
  den Dateinamen). Trifft das zu, war das Retrieval erfolgreich.

Das ist bewusst eine SIMPLE Metrik (passend für ein Lernprojekt). Grenzen:
  - Sie prüft nur, ob die richtige QUELLE gefunden wurde, nicht, ob die Antwort
    des LLM korrekt formuliert ist.
  - Mit nur einem Handbuch ist der Quellen-Abgleich leicht; bei mehreren
    Dokumenten wird die Metrik aussagekräftiger.
Profi-Erweiterung (im README erwähnt): Frameworks wie "RAGAS" messen zusätzlich
Faithfulness/Answer-Relevancy mithilfe eines LLM als Bewerter.

Aufruf:
    python -m eval.evaluate
"""

import json
from pathlib import Path

from src.retriever import retrieve
from src import config

TEST_FILE = Path(__file__).resolve().parent / "test_questions.json"


def run_evaluation():
    test_cases = json.loads(TEST_FILE.read_text(encoding="utf-8"))

    print("=" * 70)
    print(f"RAG-Retrieval-Evaluation  (top_k = {config.TOP_K})")
    print("=" * 70)

    hits = 0
    for case in test_cases:
        question = case["question"]
        keyword = case["expected_source_keyword"].lower()

        chunks = retrieve(question)
        retrieved_sources = [c["source"].lower() for c in chunks]

        # "Hit", wenn das erwartete Stichwort in IRGENDEINER Quelle vorkommt.
        hit = any(keyword in src for src in retrieved_sources)
        hits += int(hit)

        # Kleinste Distanz = bester Treffer (kleiner ist besser bei Cosine).
        best_distance = min((c["distance"] for c in chunks), default=None)

        status = "✅ HIT" if hit else "❌ MISS"
        print(f"\n{status}  Frage: {question}")
        print(f"   erwartetes Stichwort in Quelle: '{keyword}'")
        print(f"   gefundene Quellen: {retrieved_sources}")
        if best_distance is not None:
            print(f"   beste Distanz: {best_distance:.4f}")

    total = len(test_cases)
    score = hits / total if total else 0
    print("\n" + "=" * 70)
    print(f"Ergebnis:  Hit@{config.TOP_K} = {hits}/{total}  ({score:.0%})")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
