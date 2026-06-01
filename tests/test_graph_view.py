"""
Tests für die Graph-Visualisierung (Feature "Wissensgraph ansehen").

Aufruf (ohne zusätzliche Test-Bibliothek, bewusst schlank wie eval/):
    python -m tests.test_graph_view

Was wird geprüft?
  1. build_graph_payload(): die REINE Aufbereitungslogik (rohe Neo4j-Zeilen ->
     Knoten/Kanten fürs Frontend). Braucht weder Neo4j noch sonstige
     Abhängigkeiten und läuft daher überall sofort.
  2. API-Vertrag von GET /graph_data und GET /graph: nur, wenn FastAPI + httpx
     installiert sind (sonst wird dieser Teil sauber übersprungen). Neo4j muss
     NICHT laufen – der Endpoint gibt dann available=False zurück, und genau das
     prüfen wir mit.

Das Skript beendet sich mit Exit-Code 1, sobald ein Check fehlschlägt – so eignet
es sich auch für eine spätere CI.
"""

from src.graph_view import build_graph_payload, empty_payload


# --- kleine Test-Helfer (kein pytest nötig) ---------------------------------
_PASSED = 0


def check(name: str, condition: bool) -> None:
    global _PASSED
    if condition:
        _PASSED += 1
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}")
        raise AssertionError(name)


def _row(s, p, o, s_type="Component", o_type="Concept"):
    """Baut eine Neo4j-Ergebniszeile, wie sie graph_store.fetch_graph liefert."""
    return {
        "s_name": s.lower(), "s_disp": s, "s_type": s_type,
        "o_name": o.lower(), "o_disp": o, "o_type": o_type,
        "rel": p, "source_doc": "handbook.pdf", "chunk_index": 1,
    }


# --- Tests für die reine Logik ----------------------------------------------
def test_basic_assembly():
    print("test_basic_assembly")
    rows = [_row("Kamailio", "handles", "SIP signaling"),
            _row("Kamailio", "uses", "MySQL", o_type="Database")]
    g = build_graph_payload(rows)

    check("3 eindeutige Knoten", g["stats"]["nodes"] == 3)
    check("2 Kanten", g["stats"]["edges"] == 2)
    check("nicht abgeschnitten", g["stats"]["truncated"] is False)

    by_id = {n["id"]: n for n in g["nodes"]}
    check("Knoten per normalisierter id", "kamailio" in by_id)
    check("Label ist Originalschreibweise", by_id["kamailio"]["label"] == "Kamailio")
    check("Grad von Kamailio = 2", by_id["kamailio"]["degree"] == 2)
    check("Typ übernommen", by_id["mysql"]["type"] == "Database")
    check("Kante verweist auf ids", g["links"][0]["source"] == "kamailio")


def test_predicate_is_humanized():
    print("test_predicate_is_humanized")
    g = build_graph_payload([_row("A", "runs_on", "B")])
    check("Unterstrich -> Leerzeichen", g["links"][0]["type"] == "runs on")


def test_incomplete_rows_skipped():
    print("test_incomplete_rows_skipped")
    rows = [
        _row("Kamailio", "uses", "MySQL"),
        {"s_name": "x", "o_name": "", "rel": "uses"},      # Objekt fehlt
        {"s_name": "x", "o_name": "y", "rel": ""},          # Beziehung fehlt
    ]
    g = build_graph_payload(rows)
    check("nur die vollständige Kante zählt", g["stats"]["edges"] == 1)


def test_dedup_across_rows():
    print("test_dedup_across_rows")
    rows = [_row("Sipwise C5", "uses", "Kamailio", s_type="Platform"),
            _row("Sipwise C5", "uses", "SEMS", s_type="Platform")]
    g = build_graph_payload(rows)
    check("Sipwise C5 nur einmal", sum(n["id"] == "sipwise c5" for n in g["nodes"]) == 1)
    by_id = {n["id"]: n for n in g["nodes"]}
    check("Grad zählt beide Kanten", by_id["sipwise c5"]["degree"] == 2)


def test_max_nodes_truncation():
    print("test_max_nodes_truncation")
    # "hub" ist mit allen verbunden (hoher Grad); a..d sind Blätter.
    rows = [_row("hub", "links", n) for n in ["a", "b", "c", "d"]]
    g = build_graph_payload(rows, max_nodes=3)
    ids = {n["id"] for n in g["nodes"]}
    check("auf 3 Knoten begrenzt", g["stats"]["nodes"] == 3)
    check("truncated-Flag gesetzt", g["stats"]["truncated"] is True)
    check("der Hub (höchster Grad) bleibt erhalten", "hub" in ids)
    check("keine Kante zeigt ins Leere",
          all(l["source"] in ids and l["target"] in ids for l in g["links"]))


def test_empty_payload_shape():
    print("test_empty_payload_shape")
    e = empty_payload(available=False, message="Neo4j aus")
    check("available=False", e["available"] is False)
    check("Nachricht durchgereicht", e["message"] == "Neo4j aus")
    check("leere Listen", e["nodes"] == [] and e["links"] == [])
    check("Statistik-Schlüssel vorhanden",
          e["stats"] == {"nodes": 0, "edges": 0, "truncated": False})


# --- API-Vertrag (nur wenn FastAPI + httpx vorhanden sind) ------------------
def test_api_contract_if_available():
    print("test_api_contract_if_available")
    try:
        from fastapi.testclient import TestClient
        from src.api import app
    except Exception as exc:  # FastAPI/httpx/Abhängigkeiten nicht installiert
        print(f"  ⏭️  übersprungen (Abhängigkeiten fehlen: {exc})")
        return

    client = TestClient(app)

    # /graph_data antwortet IMMER formgleich – auch ohne Neo4j (available=False).
    res = client.get("/graph_data")
    check("/graph_data -> 200", res.status_code == 200)
    body = res.json()
    for key in ("available", "nodes", "links", "stats"):
        check(f"/graph_data enthält '{key}'", key in body)
    check("stats hat nodes/edges/truncated",
          set(body["stats"]) == {"nodes", "edges", "truncated"})

    # /graph liefert die HTML-Seite aus.
    page = client.get("/graph")
    check("/graph -> 200", page.status_code == 200)
    check("/graph ist HTML", "text/html" in page.headers.get("content-type", ""))


def main():
    tests = [
        test_basic_assembly, test_predicate_is_humanized,
        test_incomplete_rows_skipped, test_dedup_across_rows,
        test_max_nodes_truncation, test_empty_payload_shape,
        test_api_contract_if_available,
    ]
    print("=" * 60)
    print("Graph-View Tests")
    print("=" * 60)
    for t in tests:
        t()
    print("=" * 60)
    print(f"✅ {_PASSED} Checks bestanden.")
    print("=" * 60)


if __name__ == "__main__":
    main()
