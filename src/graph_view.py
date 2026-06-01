"""
GraphRAG-Erweiterung: Daten-AUFBEREITUNG für die interaktive Graph-Ansicht.

Dieses Modul enthält BEWUSST keinerlei Datenbank- oder Netzwerk-Code. Es ist
eine reine "Übersetzer"-Funktion: rein kommen rohe Kanten-Zeilen (so wie sie eine
Cypher-Abfrage liefert), raus kommt eine fertige Struktur aus KNOTEN und KANTEN,
die das Frontend (web/graph.html) direkt zeichnen kann.

WARUM ein eigenes Modul (und nicht alles in graph_store.py)?
  - graph_store.py importiert den Neo4j-Treiber. Diese reine Logik hier kommt
    ohne aus -> sie lässt sich isoliert und OHNE laufende Datenbank testen
    (siehe tests/test_graph_view.py). Das ist dasselbe "ein Modul, eine klar
    abgegrenzte Aufgabe"-Prinzip wie im restlichen Projekt.

DAS AUSGABE-FORMAT (an force-graph im Browser angelehnt):
    {
      "nodes": [ {"id", "label", "type", "degree"}, ... ],
      "links": [ {"source", "target", "type", "source_doc", "chunk_index"}, ... ],
      "stats": {"nodes", "edges", "truncated"},
    }
  - "id" ist der normalisierte Entitätsname (eindeutig, wie in Neo4j).
  - "label" ist die schöne Originalschreibweise (display) für die Anzeige.
  - "degree" (Grad) = Anzahl der Kanten an einem Knoten -> steuert die Punktgröße.
  - "source"/"target" der Kanten verweisen auf die "id" der Knoten.
"""

from __future__ import annotations


def build_graph_payload(rows: list[dict], max_nodes: int | None = None) -> dict:
    """
    Wandelt rohe Kanten-Zeilen aus Neo4j in eine Knoten-/Kanten-Struktur um.

    Erwartetes Format pro Zeile (genau das, was graph_store.fetch_graph liefert):
        {
          "s_name", "s_disp", "s_type",   # Subjekt (Start-Knoten)
          "o_name", "o_disp", "o_type",   # Objekt  (Ziel-Knoten)
          "rel",                          # Beziehung (Kanten-Beschriftung)
          "source_doc", "chunk_index",    # Herkunft des Fakts (optional)
        }

    max_nodes begrenzt die Anzahl angezeigter Knoten. Wird sie überschritten,
    behalten wir die Knoten mit dem HÖCHSTEN GRAD (die am stärksten vernetzten),
    weil die für das Verständnis des Graphen am wertvollsten sind. Kanten zu
    weggefallenen Knoten werden dann ebenfalls entfernt, damit keine "ins Leere"
    zeigenden Linien übrig bleiben.
    """
    nodes: dict[str, dict] = {}
    degree: dict[str, int] = {}
    links: list[dict] = []

    for row in rows:
        s_name = (row.get("s_name") or "").strip()
        o_name = (row.get("o_name") or "").strip()
        rel = (row.get("rel") or "").strip()
        # Unvollständige Zeilen überspringen (Subjekt, Objekt und Beziehung sind
        # für eine sinnvolle Kante alle drei nötig).
        if not s_name or not o_name or not rel:
            continue

        _ensure_node(nodes, s_name, row.get("s_disp"), row.get("s_type"))
        _ensure_node(nodes, o_name, row.get("o_disp"), row.get("o_type"))

        # Grad beider Endpunkte erhöhen (zählt, wie "wichtig"/vernetzt ein Knoten ist).
        degree[s_name] = degree.get(s_name, 0) + 1
        degree[o_name] = degree.get(o_name, 0) + 1

        links.append({
            "source": s_name,
            "target": o_name,
            "type": rel.replace("_", " "),  # "runs_on" -> "runs on" (lesbarer)
            "source_doc": row.get("source_doc"),
            "chunk_index": row.get("chunk_index"),
        })

    # Grad in die Knoten schreiben (steuert später die Punktgröße im Browser).
    for name, node in nodes.items():
        node["degree"] = degree.get(name, 0)

    truncated = False
    if max_nodes is not None and len(nodes) > max_nodes:
        # Die am stärksten vernetzten Knoten behalten.
        kept = sorted(nodes.values(), key=lambda n: n["degree"], reverse=True)[:max_nodes]
        kept_ids = {n["id"] for n in kept}
        nodes = {n["id"]: n for n in kept}
        # Nur Kanten behalten, deren BEIDE Endpunkte noch da sind.
        links = [
            link for link in links
            if link["source"] in kept_ids and link["target"] in kept_ids
        ]
        truncated = True

    return {
        "nodes": list(nodes.values()),
        "links": links,
        "stats": {
            "nodes": len(nodes),
            "edges": len(links),
            "truncated": truncated,
        },
    }


def empty_payload(available: bool = True, message: str | None = None) -> dict:
    """
    Eine leere, aber FORMGLEICHE Antwort (gleiche Schlüssel wie oben).

    Genutzt, wenn der Graph leer ist ODER Neo4j nicht erreichbar ist. So muss das
    Frontend keine Sonderfälle behandeln – es bekommt immer dieselbe Struktur,
    nur eben mit leeren Listen und passenden Flags/Hinweisen.
    """
    return {
        "available": available,
        "message": message,
        "nodes": [],
        "links": [],
        "stats": {"nodes": 0, "edges": 0, "truncated": False},
    }


def _ensure_node(nodes: dict, name: str, display: str | None, type_: str | None) -> None:
    """Legt einen Knoten an, falls er noch nicht existiert (dedupliziert per id)."""
    if name not in nodes:
        nodes[name] = {
            "id": name,
            # display ist die schöne Schreibweise; fehlt sie, nehmen wir die id.
            "label": (display or "").strip() or name,
            "type": (type_ or "Unknown").strip() or "Unknown",
        }
