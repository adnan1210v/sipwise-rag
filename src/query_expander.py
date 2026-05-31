"""
Kleine Query-Normalisierung für Deutsch/Englisch.

Warum gibt es dieses Modul?
Die Dokumente in data/ sind überwiegend Englisch. Das Embedding-Modell kann zwar
Semantik suchen, aber mit kurzen deutschen Fragen und Tippfehlern wird es
spürbar schwächer. Statt sofort die komplette Vektor-DB mit einem größeren
multilingualen Modell neu aufzubauen, machen wir einen pragmatischen ersten
Schritt: Aus einer Nutzerfrage entstehen mehrere Suchvarianten.

Beispiel:
    "was ist sipwise geanu"
    -> "was ist sipwise genau"
    -> "what is Sipwise C5"

Das ist bewusst kein perfekter Übersetzer. Es ist eine lokale, erklärbare
Verbesserung für typische Demo-/Interview-Fragen, ohne Cloud-API und ohne LLM.
"""

from __future__ import annotations

import re


_TYPO_FIXES = {
    "geanu": "genau",
    "genua": "genau",
    "gena": "genau",
    "sipwiese": "sipwise",
    "sipwize": "sipwise",
    "kamilo": "kamailio",
    "kamalio": "kamailio",
    "rtp engine": "rtpengine",
}


_PHRASE_TRANSLATIONS = [
    (r"\bwas ist\b", "what is"),
    (r"\bwas macht\b", "what does"),
    (r"\bwie funktioniert\b", "how does"),
    (r"\bwie nutzt\b", "how does"),
    (r"\bwie benutzt\b", "how does"),
    (r"\bwie verwendet\b", "how does"),
    (r"\bwie wird\b", "how is"),
    (r"\bworaus besteht\b", "which components does"),
    (r"\bwofür ist\b", "what is"),
]


_WORD_TRANSLATIONS = {
    "genau": "",
    "eigentlich": "",
    "bitte": "",
    "erklär": "explain",
    "erklaer": "explain",
    "erkläre": "explain",
    "erklaere": "explain",
    "nutzt": "uses",
    "benutzt": "uses",
    "verwendet": "uses",
    "konfiguriere": "configure",
    "konfiguriert": "configured",
    "einrichten": "configure",
    "datenbank": "database",
    "plattform": "platform",
    "komponenten": "components",
    "bestandteile": "components",
    "zusammenhang": "connection",
    "verbunden": "connected",
    "anruf": "call",
    "anrufe": "calls",
    "telefonie": "telephony",
    "signalisierung": "signaling",
    "medien": "media",
}


_PRODUCT_CASING = {
    "sipwise": "Sipwise",
    "c5": "C5",
    "c4": "C4",
    "kamailio": "Kamailio",
    "rtpengine": "RTPengine",
    "sems": "SEMS",
    "mysql": "MySQL",
    "sip": "SIP",
    "rtp": "RTP",
    "ssh": "SSH",
}


def expand_query(question: str, max_variants: int = 4) -> list[str]:
    """
    Erzeugt Suchvarianten für eine Frage.

    Die Reihenfolge ist wichtig: zuerst die Originalfrage, dann vorsichtig
    normalisierte Varianten. Beim Zusammenführen der Treffer gewinnt später der
    beste Treffer pro Dokument-Chunk.
    """
    variants: list[str] = []
    original = _normalize_spaces(question)
    corrected = _fix_typos(original)
    translated = _translate_lightweight(corrected)
    lowered = corrected.lower()
    is_sipwise_what_is = "sipwise" in lowered and _looks_like_what_is_question(lowered)

    # Wenn wir einen klaren Tippfehler korrigieren konnten, suchen wir nicht
    # zusätzlich nach der fehlerhaften Originalform. Die Originalfrage bleibt
    # später im Prompt erhalten, aber fürs Retrieval wäre sie nur Rauschen.
    if corrected.lower() == original.lower():
        _add_unique(variants, original)
    else:
        _add_unique(variants, corrected)

    # Häufiger Demo-Fall: Nutzer fragt nur "was ist Sipwise", meint im Projekt
    # aber meist das Produkt Sipwise C5. Diese Variante zieht die richtigen
    # Produkt-Beschreibungs-Chunks nach oben, ohne die Originalfrage zu löschen.
    if is_sipwise_what_is:
        _add_unique(variants, "What is Sipwise C5?")

    # "what is Sipwise" ist für dieses Projekt zu breit: Es findet auch alte C4-
    # Fact-Sheets. In diesem Spezialfall reicht die präzisere C5-Variante oben.
    if not (is_sipwise_what_is and translated.lower() == "what is sipwise"):
        _add_unique(variants, translated)

    # Bei Fragen zu Nutzung/Verbindungen hilft eine explizite englische Form,
    # weil die Doku technische Beziehungen meist auf Englisch beschreibt.
    if "kamailio" in lowered and "sipwise" in lowered:
        _add_unique(variants, "How does Sipwise C5 use Kamailio?")

    return variants[:max_variants]


def _fix_typos(text: str) -> str:
    fixed = text
    for typo, replacement in _TYPO_FIXES.items():
        fixed = re.sub(rf"\b{re.escape(typo)}\b", replacement, fixed, flags=re.I)
    return _normalize_spaces(fixed)


def _translate_lightweight(text: str) -> str:
    translated = text.lower()

    for pattern, replacement in _PHRASE_TRANSLATIONS:
        translated = re.sub(pattern, replacement, translated, flags=re.I)

    words = []
    for raw_word in re.split(r"(\W+)", translated):
        key = raw_word.lower()
        words.append(_WORD_TRANSLATIONS.get(key, raw_word))

    translated = "".join(words)
    translated = _normalize_spaces(translated)
    translated = _restore_product_casing(translated)
    return translated


def _restore_product_casing(text: str) -> str:
    restored = text
    for lower, proper in _PRODUCT_CASING.items():
        restored = re.sub(rf"\b{re.escape(lower)}\b", proper, restored, flags=re.I)
    return restored


def _looks_like_what_is_question(lowered: str) -> bool:
    return (
        "was ist" in lowered
        or "what is" in lowered
        or "erklär" in lowered
        or "erklaer" in lowered
    )


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _add_unique(items: list[str], value: str) -> None:
    value = _normalize_spaces(value)
    if value and value.lower() not in {item.lower() for item in items}:
        items.append(value)
