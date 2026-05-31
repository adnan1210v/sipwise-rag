"""
Schritt 6 der Pipeline: ANTWORT generieren (das "G" in RAG).

Wir geben dem lokalen LLM (über Ollama) die abgerufenen Chunks als Kontext und
bitten es, NUR auf Basis dieses Kontexts zu antworten. Das ist der Kern von
RAG: Das Modell soll nicht "frei fantasieren" (halluzinieren), sondern sich auf
die mitgelieferten Dokumente stützen.
"""

import re
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
    "4. Antworte exakt in der Antwortsprache, die im Prompt angegeben ist "
    "(Deutsch oder Englisch).\n"
    "5. Wenn die Nutzerfrage Tippfehler enthält oder eine normalisierte Frage "
    "angegeben ist, beantworte die offensichtlich gemeinte Frage. Fakten müssen "
    "trotzdem aus dem Kontext kommen.\n"
    "6. Übersetze technische Produktbegriffe nicht krampfhaft. Begriffe wie "
    "SIP-based Class 5 VoIP soft-switch platform dürfen auf Englisch stehen, "
    "wenn eine deutsche Übersetzung unnatürlich wäre. Übersetze soft-switch "
    "auf Deutsch höchstens als Softswitch, nie wörtlich als Weiche/Kabel.\n"
    "7. Bei allgemeinen 'Was ist ...?'-Fragen: Gib zuerst eine kurze Definition "
    "und danach nur die wichtigsten Eigenschaften. Nenne Detailthemen wie "
    "STIR/SHAKEN nur, wenn die Frage danach fragt.\n"
    "8. Steht die Antwort gar nicht im Kontext, sage ehrlich, dass die "
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
    query_variants = _query_variants_from_chunks(question, chunks)
    answer_language = detect_answer_language(question)

    normalized_question = query_variants[0] if query_variants else None

    if query_variants:
        query_hint = (
            "Zur Einordnung: Für die Suche wurden diese normalisierten Varianten "
            "der Nutzerfrage verwendet. Nutze sie, um Sprache und Tippfehler zu "
            "verstehen; die Fakten müssen weiterhin aus dem Kontext stammen:\n"
            + "\n".join(f"- {variant}" for variant in query_variants)
            + "\n\n"
        )
    else:
        query_hint = ""

    return (
        f"{query_hint}"
        f"Kontext:\n{context}\n\n"
        f"Originalfrage: {question}\n"
        + (f"Normalisierte Frage: {normalized_question}\n" if normalized_question else "")
        + f"Antwortsprache: {answer_language}\n"
        + "\n"
        f"Antwort (nur basierend auf dem Kontext oben):"
    )


def _query_variants_from_chunks(question: str, chunks: list[dict]) -> list[str]:
    """Sammelt die Suchvarianten, die tatsächlich Treffer geliefert haben."""
    variants: list[str] = []
    for chunk in chunks:
        variant = chunk.get("query_variant")
        if (
            variant
            and variant.lower() != question.lower()
            and variant.lower() not in {item.lower() for item in variants}
        ):
            variants.append(variant)
    return variants[:3]


def detect_answer_language(question: str) -> str:
    """
    Simple Heuristik reicht hier: Die App ist für Deutsch/Englisch gedacht.
    Wenn typische deutsche Wörter vorkommen, antworten wir Deutsch, sonst
    Englisch. So kann das deutsche System-Prompt englische Fragen nicht mehr
    versehentlich "eindeutschen".
    """
    lowered = question.lower()
    german_markers = {
        "was",
        "wie",
        "wieso",
        "warum",
        "wofür",
        "wofuer",
        "erklär",
        "erklaer",
        "nutzt",
        "benutzt",
        "verwendet",
        "datenbank",
        "genau",
    }
    words = set(re.findall(r"\w+", lowered))
    if words & german_markers:
        return "Deutsch"
    return "English"


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Schickt Kontext + Frage an das lokale LLM und gibt die Antwort zurück.

    ollama.chat() spricht mit dem lokalen Ollama-Dienst. Das Modell muss vorher
    einmal heruntergeladen worden sein:  ollama pull llama3.2:3b
    """
    prompt = _build_prompt(question, chunks)
    answer_language = detect_answer_language(question)
    language_instruction = (
        f"\n\nAktuelle Antwortsprache: {answer_language}. "
        f"Du MUSST die Antwort in {answer_language} schreiben."
    )

    client = ollama.Client(
        host=config.OLLAMA_HOST,
        timeout=config.OLLAMA_TIMEOUT_SECONDS,
    )
    response = client.chat(
        model=config.LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + language_instruction},
            {"role": "user", "content": prompt},
        ],
        # temperature niedrig -> faktentreue, weniger "kreative" Antworten.
        options={"temperature": 0.1},
    )
    return polish_known_model_artifacts(response["message"]["content"])


def polish_known_model_artifacts(answer: str) -> str:
    """
    Kleine Glättung bekannter Übersetzungsfehler lokaler Kleinstmodelle.

    Das 3B-Modell übersetzt "soft-switch" manchmal wörtlich als "Weiche/Kabel".
    Das ist fachlich missverständlich; "Softswitch" ist im Deutschen der
    etablierte technische Begriff.
    """
    replacements = {
        "VoIP-Weiche/Kabel-Plattform": "VoIP-Softswitch-Plattform",
        "VoIP-Weicheleitplattform": "VoIP-Softswitch-Plattform",
        "VoIP-Weichspulsmittelplattform": "VoIP-Softswitch-Plattform",
        "VoIP-Weichekabel": "VoIP-Softswitch",
        "Class 5-Weiche": "Class-5-Softswitch",
        "Stirim/SHAKEN": "STIR/SHAKEN",
    }
    for wrong, right in replacements.items():
        answer = answer.replace(wrong, right)
    return answer
