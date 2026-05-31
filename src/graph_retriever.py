"""
GraphRAG-Schritt: RETRIEVAL aus dem Knowledge Graph (das "Suchen" im Graph).

Während das Vektor-Retrieval (retriever.py) nach ÄHNLICHEN TEXTEN sucht, sucht
dieses Modul nach VERBUNDENEN FAKTEN. Das ist der eigentliche Mehrwert von
GraphRAG:
  - Vektor-Suche: "Welche Textstellen klingen wie meine Frage?"
  - Graph-Suche:  "Welche Dinge hängen mit den Begriffen meiner Frage zusammen?"

------------------------------------------------------------------------------
DIE STRATEGIE IN ZWEI SCHRITTEN
------------------------------------------------------------------------------
1. SAAT-KNOTEN finden ("seed entities"):
   Aus der Frage ziehen wir Stichwörter und suchen Entitäten im Graph, deren Name
   dazu passt. Das sind unsere Einstiegspunkte ("wo im Netz fangen wir an?").

2. BEZIEHUNGEN FOLGEN ("graph traversal"):
   Von jedem Saat-Knoten aus sammeln wir die direkt verbundenen Fakten (die
   Nachbarn – eine "1-Hop"-Umgebung). Diese Fakten werden später dem LLM als
   strukturierter Kontext mitgegeben.

WARUM nur 1 Hop (direkte Nachbarn)? Mehr Hops = exponentiell mehr Fakten = das
kleine LLM wird überflutet und langsamer. Für unsere Doku-Fragen liefern direkte
Nachbarn die relevanten Zusammenhänge. (Trade-off bewusst – im README erklärt.)
"""

import re

from . import config
from .query_expander import expand_query
from .graph_store import run_read


# Sehr häufige Wörter, die als Such-Stichwort nutzlos sind (sie kämen in fast
# jedem Knoten vor und würden die Suche verwässern). Bewusst klein gehalten –
# eine simple, nachvollziehbare "Stoppwortliste" statt einer großen Bibliothek.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "what", "how", "does", "do", "can", "which", "who",
    "with", "by", "as", "at", "be", "this", "that", "it", "its", "from",
    "use", "uses", "used", "using", "connected", "components", "component",
    "sipwise",  # zu generisch: trifft Firma, C4, C5, Mailadressen etc.
    "wie", "was", "der", "die", "das", "und", "oder", "ist", "sind", "ein",
    "eine", "von", "zu", "im", "in", "auf", "für", "mit", "welche", "wer",
    "nutzt", "benutzt", "verwendet", "genau", "komponenten", "verbunden",
}
_SHORT_TECH_TERMS = {"c4", "c5"}


def _keywords(question: str) -> list[str]:
    """
    Zerlegt die Frage in brauchbare Stichwörter (klein, ohne Stoppwörter, >=3 Zeichen).

    Beispiel: "How does Kamailio handle SIP?" -> ["kamailio", "handle", "sip"]
    Bewusst einfach gehalten (nur Wort-Trennung + Filter), damit du jede Zeile
    verstehst. Profi-Variante wäre Named-Entity-Recognition.
    """
    # \w+ findet zusammenhängende Wort-Zeichen; alles klein für robusten Vergleich.
    words = re.findall(r"\w+", question.lower())
    return [
        w for w in words
        if (len(w) >= 3 or w in _SHORT_TECH_TERMS) and w not in _STOPWORDS
    ]


def find_seed_entities(question: str, limit: int | None = None) -> list[dict]:
    """
    Findet Saat-Knoten: Entitäten, deren Name eines der Frage-Stichwörter enthält.

    CYPHER ERKLÄRT:
      UNWIND $keywords AS kw
        -> macht aus der Stichwortliste einzelne Zeilen (eine pro Stichwort),
           damit wir gegen JEDES Stichwort matchen können.
      MATCH (e:Entity)
        -> gehe alle Entity-Knoten durch.
      WHERE e.name CONTAINS kw
        -> behalte die, deren (normalisierter) Name das Stichwort als Teilstring
           enthält. CONTAINS = "kommt darin vor" (einfacher Textvergleich).
      WITH e, count(*) AS hits
        -> zähle, mit wie vielen Stichwörtern ein Knoten matcht (mehr = relevanter).
      RETURN ... ORDER BY hits DESC LIMIT $limit
        -> die relevantesten zuerst, aber höchstens $limit Stück.
    """
    if limit is None:
        limit = config.GRAPH_SEED_ENTITIES

    questions = (
        expand_query(question, max_variants=config.QUERY_EXPANSION_MAX_VARIANTS)
        if config.QUERY_EXPANSION_ENABLED
        else [question]
    )
    keywords = []
    for q in questions:
        for keyword in _keywords(q):
            if keyword not in keywords:
                keywords.append(keyword)
    if not keywords:
        return []

    rows = run_read(
        """
        UNWIND $keywords AS kw
        MATCH (e:Entity)
        WHERE e.name CONTAINS kw
        WITH e, count(*) AS hits,
             CASE WHEN e.name IN $keywords THEN 1 ELSE 0 END AS exact
        RETURN e.name AS name, e.display AS display, e.type AS type, hits
        ORDER BY exact DESC, hits DESC, name ASC
        LIMIT $limit
        """,
        keywords=keywords,
        limit=limit,
    )
    return rows


def get_connected_facts(entity_names: list[str], limit: int | None = None) -> list[dict]:
    """
    Sammelt die direkt verbundenen Fakten (1 Hop) rund um die Saat-Knoten.

    CYPHER ERKLÄRT:
      MATCH (e:Entity)-[r:REL]-(other:Entity)
        -> finde Beziehungen, an denen einer unserer Saat-Knoten hängt.
           Achtung: KEIN Pfeil (-[r:REL]- statt ->) bedeutet "in BEIDE
           Richtungen". Wir wollen ja auch Fakten, in denen unser Knoten das
           OBJEKT ist (z. B. "X uses <Saat-Knoten>").
      WHERE e.name IN $names
        -> nur Beziehungen rund um unsere Saat-Knoten.
      RETURN startNode(r)... , r.type, endNode(r)...
        -> wir geben den Fakt in seiner ECHTEN Richtung zurück (Subjekt ->
           Objekt), egal von welcher Seite wir ihn gefunden haben. startNode()/
           endNode() liefern die tatsächlichen Quell-/Ziel-Knoten der Kante.
      DISTINCT + LIMIT
        -> keine Dubletten, und die Menge begrenzen (sonst Prompt-Explosion).
    """
    if limit is None:
        limit = config.GRAPH_MAX_FACTS
    if not entity_names:
        return []

    rows = run_read(
        """
        MATCH (e:Entity)-[r:REL]-(other:Entity)
        WHERE e.name IN $names
        WITH DISTINCT r
        RETURN startNode(r).display AS subject,
               r.type             AS predicate,
               endNode(r).display AS object,
               r.source_doc       AS source_doc,
               r.chunk_index      AS chunk_index
        LIMIT $limit
        """,
        names=entity_names,
        limit=limit,
    )
    return rows


def retrieve_graph_context(question: str) -> dict:
    """
    Haupt-Funktion: Frage -> Saat-Knoten -> verbundene Fakten.

    Rückgabe:
        {
          "seeds": [ {name, display, type, hits}, ... ],   # Einstiegspunkte
          "facts": [ {subject, predicate, object, source_doc, chunk_index}, ... ]
        }
    Diese Fakten formt der Generator/Hybrid-Retriever später in Text-Sätze um.
    """
    seeds = find_seed_entities(question)
    seed_names = [s["name"] for s in seeds]
    facts = get_connected_facts(seed_names)
    return {"seeds": seeds, "facts": facts}


def facts_to_text(facts: list[dict]) -> str:
    """
    Wandelt strukturierte Fakten in einfache Sätze für den LLM-Prompt um.

    Aus  {subject:"Kamailio", predicate:"handles", object:"SIP signaling"}
    wird "Kamailio handles SIP signaling."
    WARUM Sätze statt JSON? Ein LLM versteht natürliche Sprache am besten; kurze
    Fakten-Sätze sind für das Modell leichter zu verwerten als rohe Datenstrukturen.
    """
    lines = []
    for f in facts:
        # predicate ist intern mit Unterstrichen gespeichert (z. B. "runs_on")
        # -> für die Lesbarkeit wieder zu Leerzeichen.
        predicate = (f.get("predicate") or "").replace("_", " ")
        lines.append(f"- {f['subject']} {predicate} {f['object']}.")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Schnelltest:  python -m src.graph_retriever "How does Kamailio work?"
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    frage = " ".join(sys.argv[1:]) or "What is Kamailio?"
    print(f"❓ Frage: {frage}\n")

    result = retrieve_graph_context(frage)

    print(f"🌱 Saat-Knoten ({len(result['seeds'])}):")
    for s in result["seeds"]:
        print(f"   • {s['display']}  (Typ: {s['type']}, Treffer: {s['hits']})")

    print(f"\n🔗 Verbundene Fakten ({len(result['facts'])}):")
    print(facts_to_text(result["facts"]) or "   (keine gefunden)")

    from .graph_store import close_driver
    close_driver()
