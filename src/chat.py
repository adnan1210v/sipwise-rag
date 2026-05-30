"""
Interaktiver CHAT im Terminal.

Einmal starten -> dann einfach Fragen tippen und Enter drücken, beliebig oft.
Kein curl, keine Anführungszeichen, kein erneutes Tippen des Befehls.

Aufruf:
    python -m src.chat

Beenden: 'exit' oder 'quit' tippen (oder Strg+C).
"""

from .pipeline import answer_question


def main():
    print("=" * 60)
    print("  Sipwise C5 — RAG Chat (lokal & offline)")
    print("  Tippe deine Frage und drücke Enter.")
    print("  Beenden mit 'exit' oder 'quit'.")
    print("=" * 60)

    while True:
        # input() wartet, bis du etwas tippst und Enter drückst.
        try:
            question = input("\n❓ Frage> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Strg+C oder Strg+D -> sauber beenden statt hässlicher Fehler.
            print("\nTschüss! 👋")
            break

        if not question:
            continue  # leere Eingabe -> einfach nochmal fragen
        if question.lower() in {"exit", "quit", "q"}:
            print("Tschüss! 👋")
            break

        print("… denke nach …")
        result = answer_question(question)

        print("\n💬 Antwort:")
        print(result["answer"])
        print("\n📚 Quellen:")
        for s in result["sources"]:
            print(f"   • {s['source']}  (Chunk {s['chunk_index']}, "
                  f"Distanz {s['distance']})")


if __name__ == "__main__":
    main()
