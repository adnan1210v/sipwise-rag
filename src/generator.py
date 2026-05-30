"""
Schritt 6 der Pipeline: ANTWORT generieren (das "G" in RAG).

Wir geben dem lokalen LLM (über Ollama) die abgerufenen Chunks als Kontext und
bitten es, NUR auf Basis dieses Kontexts zu antworten. Das ist der Kern von
RAG: Das Modell soll nicht "frei fantasieren" (halluzinieren), sondern sich auf
die mitgelieferten Dokumente stützen.
"""

import ollama
from . import config

# Das "System-Prompt" gibt dem Modell seine Rolle und die wichtigsten Regeln.
# Designentscheidung: Es soll sich an den Kontext halten (gegen Halluzinationen),
# ABER ausdrücklich vereinfachen/umformulieren/erklären DÜRFEN. Sonst deutet ein
# kleines Modell die strenge Regel als "ich darf gar nichts umformen" und
# verweigert z. B. eine Bitte wie "erkläre es für einen Anfänger".
SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Assistent für die technische Dokumentation von "
    "Sipwise C5.\n"
    "Regeln:\n"
    "1. Stütze deine FAKTEN ausschließlich auf den bereitgestellten Kontext. "
    "Erfinde keine Fakten, Abkürzungen oder Zahlen.\n"
    "2. Du darfst den Kontext frei umformulieren, zusammenfassen und in "
    "einfachen Worten erklären – auch für Anfänger/Einsteiger, wenn darum "
    "gebeten wird. Vereinfachen ist erlaubt, solange es faktisch korrekt bleibt.\n"
    "3. Wenn eine Abkürzung im Kontext nicht erklärt wird, rate sie NICHT, "
    "sondern lass sie weg oder sag, dass sie nicht erläutert ist.\n"
    "4. Antworte in derselben Sprache wie die Frage.\n"
    "5. Steht die Antwort gar nicht im Kontext, sage ehrlich, dass die "
    "Dokumentation dazu nichts hergibt."
)


def _build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Baut den eigentlichen Prompt: nummerierter Kontext + die Frage.

    Wir nummerieren die Kontext-Blöcke und nennen ihre Quelle. So kann das
    Modell sich darauf beziehen und wir behalten Nachvollziehbarkeit.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Kontext {i} | Quelle: {chunk['source']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    return (
        f"Kontext:\n{context}\n\n"
        f"Frage: {question}\n\n"
        f"Antwort (nur basierend auf dem Kontext oben):"
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Schickt Kontext + Frage an das lokale LLM und gibt die Antwort zurück.

    ollama.chat() spricht mit dem lokalen Ollama-Dienst. Das Modell muss vorher
    einmal heruntergeladen worden sein:  ollama pull llama3.2:3b
    """
    prompt = _build_prompt(question, chunks)

    response = ollama.chat(
        model=config.LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # temperature niedrig -> faktentreue, weniger "kreative" Antworten.
        options={"temperature": 0.1},
    )
    return response["message"]["content"]
