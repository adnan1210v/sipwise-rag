"""
Kleines KOMMANDOZEILEN-Tool, um Fragen ohne Web-Server zu stellen.

Praktisch zum schnellen Testen der Pipeline.

Aufruf:
    python -m src.ask "How do I configure SIP peering?"
"""

import sys
from .pipeline import answer_question


def main():
    if len(sys.argv) < 2:
        print('Benutzung: python -m src.ask "Deine Frage hier"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\n❓ Frage: {question}\n")
    print("… denke nach (lokales LLM + Retrieval) …\n")

    result = answer_question(question)

    print("💬 Antwort:")
    print(result["answer"])
    print("\n📚 Quellen:")
    for s in result["sources"]:
        print(f"   • {s['source']}  (Chunk {s['chunk_index']}, "
              f"Distanz {s['distance']})")


if __name__ == "__main__":
    main()
