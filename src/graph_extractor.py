"""
GraphRAG-Schritt: ENTITÄTEN & BEZIEHUNGEN aus Text extrahieren (das "Graph" in
GraphRAG). Dieses Modul verwandelt Fließtext in strukturierte TRIPLES.

------------------------------------------------------------------------------
WAS IST EIN TRIPLE?
------------------------------------------------------------------------------
Ein Triple ist ein Fakt in der Form  (Subjekt) —[Beziehung]→ (Objekt), z. B.:
    (Kamailio) —[handles]→ (SIP signaling)
    (Sipwise C5) —[uses]→ (MySQL)
Diese drei Teile sind genau das, was ein Graph speichert: zwei KNOTEN (Subjekt,
Objekt) und eine KANTE (Beziehung) dazwischen. Wenn wir das für viele Sätze tun,
entsteht ein ganzes NETZ aus verbundenen Fakten – der Knowledge Graph.

------------------------------------------------------------------------------
WIE EXTRAHIEREN WIR? — Das LLM als "Leser"
------------------------------------------------------------------------------
Menschen erkennen solche Fakten beim Lesen automatisch. Ein LLM kann das auch –
wir müssen es nur klar anweisen. Die KERNIDEE unseres Prompt-Schemas:
  1. Wir geben dem LLM einen Textabschnitt.
  2. Wir bitten es, NUR Fakten herauszuziehen, die WIRKLICH im Text stehen
     (kein Raten – sonst halluziniert der Graph).
  3. Wir verlangen die Antwort als REINES JSON in einem festen Format. Strukturierte
     Ausgabe ist viel leichter und zuverlässiger weiterzuverarbeiten als Fließtext.

Format pro Triple (immer dieselben 5 Felder):
    {
      "subject":      "Kamailio",        # WER/WAS handelt
      "subject_type": "Component",       # grobe Kategorie des Subjekts
      "predicate":    "handles",         # die Beziehung (ein kurzes Verb)
      "object":       "SIP signaling",   # WORAUF bezieht es sich
      "object_type":  "Concept"          # grobe Kategorie des Objekts
    }

WARUM auch die "types" (Kategorien)? Sie machen den Graphen später aussagekräftiger
(man sieht: ist das eine Komponente, ein Protokoll, ein Konzept?) und helfen beim
Filtern. Wir halten die Typen bewusst frei/grob – ein kleines LLM soll sich nicht
an einer starren Liste verschlucken.

------------------------------------------------------------------------------
RAM-/GESCHWINDIGKEITS-TRADE-OFF (wichtig fürs Interview):
------------------------------------------------------------------------------
Die Extraktion ruft das LLM EINMAL PRO CHUNK auf. Das ist der mit Abstand
langsamste, teuerste Schritt von GraphRAG. Auf einem 8-GB-Mac bedeutet das:
  - Welches Modell? -> config.EXTRACTION_MODEL_NAME (3b genauer, 1b schneller).
  - Wie viele Chunks? -> config.MAX_CHUNKS_FOR_GRAPH begrenzt die Anzahl.
  - temperature=0 -> wir wollen FAKTEN, keine Kreativität.
"""

import json
import ollama

from . import config


# Der System-Prompt gibt dem LLM seine ROLLE und die strengen Regeln. Hier ist
# Strenge richtig (anders als beim Antworten-Generator): Bei der Extraktion soll
# das Modell NICHT vereinfachen oder interpretieren, sondern nur sauber das
# herausziehen, was dasteht.
EXTRACTION_SYSTEM_PROMPT = (
    "Du bist ein präzises Extraktionswerkzeug für Wissensgraphen. "
    "Deine Aufgabe: aus technischem Text die enthaltenen Fakten als Triples "
    "(Subjekt, Beziehung, Objekt) herausziehen.\n"
    "Strikte Regeln:\n"
    "1. Extrahiere NUR Fakten, die wörtlich oder klar sinngemäß im Text stehen. "
    "Erfinde nichts.\n"
    "2. Subjekt und Objekt sind kurze Begriffe (Eigennamen, Komponenten, "
    "Konzepte), keine ganzen Sätze.\n"
    "3. Die Beziehung (predicate) ist ein kurzes englisches Verb/Verbphrase in "
    "Kleinbuchstaben, z. B. 'uses', 'runs on', 'handles', 'is part of'.\n"
    "4. Antworte AUSSCHLIESSLICH mit gültigem JSON (einer Liste). Kein Fließtext, "
    "keine Erklärungen, kein Markdown.\n"
    "5. Gibt der Text keine klaren Fakten her, antworte mit einer leeren Liste []."
)


def _build_extraction_prompt(text: str) -> str:
    """
    Baut den eigentlichen Extraktions-Prompt: Anleitung + Beispiel + der Text.

    WARUM ein Beispiel (sog. "Few-Shot")? Ein kleines LLM hält sich viel
    zuverlässiger an ein Format, wenn es EINMAL sieht, wie das Ergebnis aussehen
    soll. Das Beispiel ist unser "Musterlösung"-Trick gegen kaputtes JSON.
    """
    return (
        "Extrahiere die Fakten aus dem folgenden Text als JSON-Liste von Triples.\n"
        "Jedes Triple hat exakt diese Felder: subject, subject_type, predicate, "
        "object, object_type.\n\n"
        "BEISPIEL\n"
        "Text: \"Kamailio is a SIP server used by the Sipwise C5 platform to "
        "handle SIP signaling. It stores data in a MySQL database.\"\n"
        "Antwort:\n"
        "[\n"
        '  {"subject": "Kamailio", "subject_type": "Component", '
        '"predicate": "is a", "object": "SIP server", "object_type": "Concept"},\n'
        '  {"subject": "Sipwise C5", "subject_type": "Platform", '
        '"predicate": "uses", "object": "Kamailio", "object_type": "Component"},\n'
        '  {"subject": "Kamailio", "subject_type": "Component", '
        '"predicate": "handles", "object": "SIP signaling", "object_type": "Concept"},\n'
        '  {"subject": "Kamailio", "subject_type": "Component", '
        '"predicate": "stores data in", "object": "MySQL", "object_type": "Database"}\n'
        "]\n\n"
        "JETZT DU\n"
        f"Text: \"{text}\"\n"
        "Antwort:"
    )


def _parse_triples(raw: str) -> list[dict]:
    """
    Holt die JSON-Liste aus der LLM-Antwort – robust gegen kleine Ausreißer.

    PROBLEM: Auch wenn wir reines JSON verlangen, packt ein kleines LLM manchmal
    noch Text drumherum (z. B. "Hier ist das JSON: [...]"). Statt sofort
    aufzugeben, schneiden wir den Bereich von der ersten '[' bis zur letzten ']'
    heraus und versuchen, NUR das zu parsen. Das ist eine simple, aber sehr
    wirksame "Selbstheilung".
    """
    raw = raw.strip()

    # Manche Modelle umschließen JSON mit ```json ... ``` -> entfernen.
    if raw.startswith("```"):
        raw = raw.strip("`")
        # nach dem Entfernen kann am Anfang noch "json" stehen
        if raw.lstrip().lower().startswith("json"):
            raw = raw.lstrip()[4:]

    # Den Bereich der eigentlichen Liste herausschneiden.
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []  # gar keine Liste gefunden -> nichts extrahiert

    snippet = raw[start:end + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        # Kaputtes JSON -> lieber diesen Chunk überspringen als abstürzen.
        return []

    # Sicherstellen, dass es wirklich eine Liste von Dicts ist.
    if not isinstance(data, list):
        return []
    return [t for t in data if isinstance(t, dict)]


def extract_triples(text: str) -> list[dict]:
    """
    Extrahiert Triples aus EINEM Textstück (Chunk) mithilfe des lokalen LLM.

    Rückgabe: Liste von Triple-Dicts (kann leer sein). Das Format passt exakt zu
    graph_store.write_triples().
    """
    prompt = _build_extraction_prompt(text)

    response = ollama.chat(
        model=config.EXTRACTION_MODEL_NAME,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # temperature=0 -> möglichst deterministisch & faktentreu. Bei Extraktion
        # wollen wir keine "Kreativität", sondern immer dieselbe saubere Struktur.
        options={"temperature": 0},
    )
    raw = response["message"]["content"]
    return _parse_triples(raw)


# -----------------------------------------------------------------------------
# Schnelltest an EINEM Beispieltext:  python -m src.graph_extractor
# -----------------------------------------------------------------------------
# Praktisch, um zu sehen, WIE der Extraktor arbeitet, bevor man hunderte Chunks
# verarbeitet. So verstehst du das Verhalten an einem überschaubaren Beispiel.
if __name__ == "__main__":
    beispiel = (
        "The Sipwise C5 platform uses Kamailio as its SIP proxy. "
        "Kamailio communicates with the SEMS application server for media "
        "handling. All call data records are stored in a MySQL database."
    )
    print("Beispieltext:\n", beispiel, "\n")
    print("Extrahiere Triples (das lokale LLM denkt nach) ...\n")
    triples = extract_triples(beispiel)
    print(f"-> {len(triples)} Triple(s) extrahiert:\n")
    for t in triples:
        print(f"   ({t.get('subject')}) -[{t.get('predicate')}]-> "
              f"({t.get('object')})")
