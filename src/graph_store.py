"""
GraphRAG-Schritt: SPEICHERN & ABFRAGEN im Knowledge Graph (Neo4j).

Dieses Modul ist das Gegenstück zu vector_store.py – nur eben für den GRAPH
statt für die Vektor-DB. Es kapselt ALLES, was direkt mit Neo4j spricht:
  - Verbindung aufbauen/schließen (Treiber als Singleton, wie beim Embedding-Modell)
  - das Schema anlegen (eine Eindeutigkeits-Regel für Entitäten)
  - Triples (Knoten + Kanten) schreiben
  - den Graphen leeren (für sauberes Neu-Befüllen, analog reset_collection())
  - lesende Cypher-Abfragen ausführen (nutzt später das Graph-Retrieval)
  - Statistiken/Health-Check (zum Prüfen, ob alles läuft)

------------------------------------------------------------------------------
DAS DATENMODELL (wichtig zum Verstehen, auch fürs Interview):
------------------------------------------------------------------------------
Ein Knowledge Graph besteht aus KNOTEN (nodes) und KANTEN (relationships).

  KNOTEN:  (:Entity {name, display, type})
    - "Entity" ist das "Label" (wie ein Tabellenname / eine Kategorie).
    - name    = normalisierter Schlüssel (klein geschrieben) -> verhindert, dass
                "Kamailio" und "kamailio" zwei Knoten werden.
    - display = die Originalschreibweise (schön für die Ausgabe).
    - type    = grobe Kategorie, die das LLM rät (z. B. "Component", "Protocol").

  KANTEN:  (:Entity)-[:REL {type, source_doc, chunk_index}]->(:Entity)
    - Wir nutzen EINEN Kanten-Typ ":REL" und speichern die eigentliche Beziehung
      als Eigenschaft "type" (z. B. "runs_on", "uses").
    - WARUM nicht echte Kanten-Typen wie :RUNS_ON? Neo4j-Kantentypen sind fest
      im Schema; das LLM erfindet aber beliebige Beziehungen. Dynamische
      Kantentypen bräuchten die APOC-Erweiterung (mehr RAM/Komplexität). Ein
      Kanten-Typ + "type"-Eigenschaft ist einfacher, gut abfragbar und reicht
      für unser Lernprojekt. (Trade-off bewusst dokumentiert.)
    - source_doc / chunk_index = WOHER stammt dieser Fakt? -> Nachvollziehbarkeit
      (dasselbe "grounding"-Prinzip wie bei den Quellenangaben im Vector-RAG).
------------------------------------------------------------------------------
"""

from neo4j import GraphDatabase
from . import config


# Modul-globale Variable: der Neo4j-"Treiber" (die Verbindung) wird einmal
# erzeugt und wiederverwendet. Den Treiber bei jeder Abfrage neu aufzubauen
# wäre langsam und würde unnötig viele Verbindungen öffnen.
_driver = None


def get_driver():
    """
    Liefert den Neo4j-Treiber (baut ihn beim ersten Aufruf auf).

    Der "Treiber" ist das Objekt, über das alle Abfragen laufen. Er verwaltet
    intern einen Pool von Verbindungen – wir müssen uns darum nicht kümmern.
    """
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
    return _driver


def close_driver():
    """
    Schließt die Verbindung sauber (z. B. am Ende eines Skripts).

    Nicht zwingend nötig (Python räumt am Programmende auf), aber sauberer Stil –
    offene Netzwerkverbindungen sollte man explizit schließen.
    """
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def ping() -> bool:
    """
    Health-Check: Können wir Neo4j erreichen UND eine Abfrage ausführen?

    Gibt True zurück, wenn eine triviale Abfrage ('RETURN 1') klappt.
    Praktisch, um nach dem Docker-Start zu prüfen, ob alles steht.
    """
    try:
        driver = get_driver()
        # verify_connectivity() prüft die Verbindung ohne eigene Abfrage,
        # zusätzlich führen wir noch eine Mini-Abfrage aus, um ganz sicher zu sein.
        driver.verify_connectivity()
        with driver.session() as session:
            # "RETURN 1" ist die simpelste Cypher-Abfrage: gib einfach die Zahl 1
            # zurück. .single() holt genau eine Ergebniszeile.
            result = session.run("RETURN 1 AS ok").single()
            return result is not None and result["ok"] == 1
    except Exception as e:
        print(f"  ⚠️  Keine Verbindung zu Neo4j: {e}")
        return False


def setup_schema():
    """
    Legt EINMALIG die Schema-Regel an: jede Entität (name) ist eindeutig.

    CYPHER ERKLÄRT:
      CREATE CONSTRAINT ... IF NOT EXISTS
        -> erstellt eine "Constraint" (Regel). IF NOT EXISTS = nur, falls noch
           nicht vorhanden (so kann man es gefahrlos mehrfach aufrufen).
      FOR (e:Entity) REQUIRE e.name IS UNIQUE
        -> "für jeden Knoten mit Label Entity muss die Eigenschaft name
           einzigartig sein".
    NUTZEN: (1) Verhindert doppelte Entitäten. (2) Neo4j legt automatisch einen
    INDEX an -> spätere MERGE/MATCH auf e.name werden blitzschnell.
    """
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.name IS UNIQUE
            """
        )


def reset_graph():
    """
    Löscht den GESAMTEN Graphen (alle Knoten + Kanten).

    Gegenstück zu reset_collection() im Vector-Store: vor einem Neu-Befüllen
    räumen wir auf, damit keine Dubletten/Altlasten entstehen.

    CYPHER ERKLÄRT:
      MATCH (n)        -> finde ALLE Knoten (kein Label = wirklich alle).
      DETACH DELETE n  -> lösche jeden Knoten SAMT seiner Kanten ("detach" löst
                          erst die Kanten, sonst würde Neo4j den Löschvorgang
                          verweigern, weil noch Kanten dranhängen).
    """
    driver = get_driver()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def write_triples(triples: list[dict], source_doc: str, chunk_index: int):
    """
    Schreibt eine Liste extrahierter Triples für EINEN Chunk in den Graphen.

    Erwartetes Format pro Triple (so liefert es der Extraktor in Phase 2):
        {
          "subject": "kamailio",        "subject_type": "Component",
          "predicate": "handles",
          "object": "sip signaling",    "object_type": "Concept",
        }

    CYPHER ERKLÄRT (das Kern-Statement unten):
      MERGE (s:Entity {name: $s_name})
        -> "finde ODER erstelle" einen Entity-Knoten mit diesem name.
           MERGE ist das Schweizer Taschenmesser gegen Duplikate: existiert der
           Knoten schon, wird er wiederverwendet; sonst neu angelegt.
      ON CREATE SET s.display = $s_disp, s.type = $s_type
        -> nur BEIM ERSTELLEN zusätzliche Eigenschaften setzen (display/type).
           Bei Wiederverwendung lassen wir die bestehenden Werte in Ruhe.
      MERGE (s)-[r:REL {type: $predicate}]->(o)
        -> finde/erstelle eine Kante dieses Beziehungstyps zwischen s und o.
           So wird derselbe Fakt nicht doppelt gespeichert.
      ON CREATE SET r.source_doc = ..., r.chunk_index = ...
        -> Herkunft des Fakts merken (für Quellenangaben).

    Trade-off: Kommt derselbe Fakt aus mehreren Chunks, behalten wir die Herkunft
    des ERSTEN Vorkommens. Für ein Lernprojekt ausreichend; produktiv könnte man
    alle Quellen in einer Liste sammeln.
    """
    driver = get_driver()
    with driver.session() as session:
        for t in triples:
            # Namen normalisieren (klein + getrimmt) -> robustes Deduplizieren.
            s_name = _normalize(t.get("subject", ""))
            o_name = _normalize(t.get("object", ""))
            predicate = (t.get("predicate") or "").strip().lower().replace(" ", "_")

            # Unvollständige Triples überspringen (das LLM liefert nicht immer
            # perfekte Daten – wir filtern Müll lieber raus, als ihn zu speichern).
            if not s_name or not o_name or not predicate:
                continue
            source_id = f"{source_doc}::{chunk_index}"

            session.run(
                """
                MERGE (s:Entity {name: $s_name})
                  ON CREATE SET s.display = $s_disp, s.type = $s_type
                MERGE (o:Entity {name: $o_name})
                  ON CREATE SET o.display = $o_disp, o.type = $o_type
                MERGE (s)-[r:REL {type: $predicate}]->(o)
                  ON CREATE SET r.source_doc = $source_doc,
                                r.chunk_index = $chunk_index,
                                r.sources = [$source_id]
                  ON MATCH SET r.sources =
                    CASE
                      WHEN r.sources IS NULL THEN [$source_id]
                      WHEN NOT $source_id IN r.sources THEN r.sources + $source_id
                      ELSE r.sources
                    END
                """,
                s_name=s_name,
                s_disp=t.get("subject", "").strip(),
                s_type=(t.get("subject_type") or "Unknown").strip(),
                o_name=o_name,
                o_disp=t.get("object", "").strip(),
                o_type=(t.get("object_type") or "Unknown").strip(),
                predicate=predicate,
                source_doc=source_doc,
                chunk_index=chunk_index,
                source_id=source_id,
            )


def run_read(cypher: str, **params) -> list[dict]:
    """
    Führt eine LESENDE Cypher-Abfrage aus und gibt die Zeilen als Dicts zurück.

    Allgemeiner Helfer, den das Graph-Retrieval (Phase 2) benutzt. **params
    werden sicher als Parameter eingesetzt (kein Zusammenbasteln von Strings ->
    schützt vor "Cypher-Injection", analog zu SQL).
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, **params)
        # .data() wandelt jede Ergebniszeile in ein normales Python-Dict.
        return result.data()


def graph_stats() -> dict:
    """
    Zählt Knoten und Kanten – nützlich zum Prüfen nach dem Befüllen.

    CYPHER ERKLÄRT:
      MATCH (n) RETURN count(n)        -> Anzahl aller Knoten.
      MATCH ()-[r]->() RETURN count(r) -> Anzahl aller Kanten.
    """
    driver = get_driver()
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return {"nodes": nodes, "relationships": rels}


def fetch_graph(max_nodes: int | None = None, max_edges: int | None = None) -> dict:
    """
    Holt den GESAMTEN Graphen (bis zu einer Obergrenze) für die Visualisierung.

    Das ist die Datenquelle der interaktiven Browser-Ansicht (web/graph.html).
    Gegenstück zu retrieve()/graph_retriever, die nur einen kleinen, fragebezogenen
    Ausschnitt holen – hier wollen wir den Graphen als Ganzes ZEIGEN.

    Ablauf:
      1. Eine einzige Cypher-Abfrage holt bis zu max_edges Kanten samt ihrer
         beiden Endknoten.
      2. graph_view.build_graph_payload() formt daraus Knoten + Kanten fürs Frontend.

    Robustheit: Ist Neo4j nicht erreichbar (Container aus), FANGEN wir den Fehler
    ab und geben eine leere, aber formgleiche Antwort mit available=False zurück.
    So bleibt die Web-UI bedienbar und kann einen freundlichen Hinweis anzeigen,
    statt mit einem 500er-Fehler abzustürzen (gleiche Haltung wie beim /health).

    CYPHER ERKLÄRT:
      MATCH (s:Entity)-[r:REL]->(o:Entity)
        -> finde alle Beziehungen samt Start- (s) und Ziel-Knoten (o).
      RETURN s.name, s.display, ... , r.type, r.source_doc, r.chunk_index
        -> alles, was das Frontend für Knoten, Kanten und Herkunft braucht.
      LIMIT $max_edges
        -> harte Obergrenze, damit ein riesiger Graph den Browser nicht überlastet.
    """
    # Import hier (nicht oben), um den Modul-Kopf schlank zu halten – graph_view
    # ist reine Logik ohne eigene Abhängigkeiten.
    from . import config
    from .graph_view import build_graph_payload, empty_payload

    if max_nodes is None:
        max_nodes = config.GRAPH_VIEW_MAX_NODES
    if max_edges is None:
        max_edges = config.GRAPH_VIEW_MAX_EDGES

    try:
        rows = run_read(
            """
            MATCH (s:Entity)-[r:REL]->(o:Entity)
            RETURN s.name    AS s_name, s.display AS s_disp, s.type AS s_type,
                   o.name    AS o_name, o.display AS o_disp, o.type AS o_type,
                   r.type    AS rel,
                   r.source_doc  AS source_doc,
                   r.chunk_index AS chunk_index
            LIMIT $max_edges
            """,
            max_edges=max_edges,
        )
    except Exception as e:
        # Neo4j aus / nicht erreichbar -> leere, formgleiche Antwort.
        return empty_payload(
            available=False,
            message=f"Neo4j ist nicht erreichbar ({e}). Läuft der Container "
                    f"(docker compose up -d) und wurde der Graph gebaut "
                    f"(python -m src.graph_ingest)?",
        )

    payload = build_graph_payload(rows, max_nodes=max_nodes)
    payload["available"] = True
    # Hinweis ergänzen, falls der Graph zwar erreichbar, aber (noch) leer ist.
    if not payload["nodes"]:
        payload["message"] = (
            "Der Wissensgraph ist leer. Baue ihn zuerst mit "
            "'python -m src.graph_ingest' (Neo4j muss laufen)."
        )
    else:
        payload["message"] = None
    return payload


def _normalize(name: str) -> str:
    """
    Vereinheitlicht einen Entitätsnamen zum eindeutigen Schlüssel.

    Schritte: trimmen, Kleinschreibung, mehrfache Leerzeichen zu einem. So
    werden "  SIP  Peering" und "sip peering" zur selben Entität – das hält den
    Graphen sauber (weniger Duplikate = aussagekräftigere Verknüpfungen).
    """
    return " ".join(name.strip().lower().split())


# -----------------------------------------------------------------------------
# Direkt ausführbar zum PRÜFEN der Verbindung:  python -m src.graph_store
# -----------------------------------------------------------------------------
# Das ist dein "Funktioniert Neo4j?"-Selbsttest nach dem Docker-Start.
if __name__ == "__main__":
    print("=" * 60)
    print("Neo4j-Verbindungstest")
    print("=" * 60)
    print(f"Verbinde mit: {config.NEO4J_URI} (User: {config.NEO4J_USER})")

    if ping():
        print("✅ Verbindung steht – Neo4j antwortet.")
        # Schema gleich mit anlegen (idempotent, schadet nicht).
        setup_schema()
        print("✅ Schema/Constraint ist eingerichtet.")
        stats = graph_stats()
        print(f"📊 Aktueller Graph: {stats['nodes']} Knoten, "
              f"{stats['relationships']} Kanten.")
        print("\nAlles bereit. Du kannst jetzt den Graphen befüllen (Phase 2).")
    else:
        print("❌ Keine Verbindung. Checkliste:")
        print("   1. Läuft der Container?   ->  docker compose ps")
        print("   2. Ist er 'healthy'?      ->  docker compose logs -f neo4j")
        print("   3. Passt das Passwort?    ->  NEO4J_AUTH in docker-compose.yml")
        print("      muss zu NEO4J_PASSWORD in src/config.py passen.")

    close_driver()
