"""
test_event_store.py -- EventStore incident queries (query / counts_by_level)
behind GET /api/events. SQLite in a temp dir, no GPU.

    python tests/test_event_store.py
    pytest  tests/test_event_store.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.event_store import EventStore                      # noqa: E402

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def _store() -> EventStore:
    s = EventStore(str(Path(tempfile.mkdtemp()) / "events.db"))
    s.log(0, "High", 61.0, 2, 0, {"breach": True}, "h1")
    time.sleep(0.01)
    s.log(1, "Critical", 92.0, 1, 1, {"weapon": True}, "h2")
    time.sleep(0.01)
    s.log(0, "Critical", 88.0, 3, 0, {}, "h3")
    return s


@case("query is newest-first with details decoded")
def _():
    rows = _store().query()
    assert [r["ev_hash"] for r in rows] == ["h3", "h2", "h1"]
    assert rows[1]["details"] == {"weapon": True}


@case("filters by camera and level combine")
def _():
    s = _store()
    assert [r["ev_hash"] for r in s.query(cam_id=0)] == ["h3", "h1"]
    assert [r["ev_hash"] for r in s.query(cam_id=0, level="Critical")] == ["h3"]


@case("limit is clamped to at least 1")
def _():
    assert len(_store().query(limit=0)) == 1


@case("counts_by_level aggregates, since filters")
def _():
    s = _store()
    assert s.counts_by_level() == {"High": 1, "Critical": 2}
    assert s.counts_by_level(since=time.time() + 60) == {}


@case("query still works after recent()/unsynced() changed row_factory")
def _():
    s = _store()
    s.recent(5)
    s.unsynced()
    rows = s.query(limit=2)
    assert isinstance(rows[0], dict) and rows[0]["ev_hash"] == "h3"


# ── schema 2: identity, severity/category split, cursor, acknowledgement ────

@case("severity and category are recorded separately")
def _():
    s = EventStore(str(Path(tempfile.mkdtemp()) / "events.db"))
    s.log(0, "Critical", 96.0, category="intrusion")
    s.log(0, "Info", 55.0, category="plate")
    c = s.counts()
    assert c["severity"] == {"Critical": 1, "Info": 1}
    assert c["category"] == {"intrusion": 1, "plate": 1}


@case("the legacy level column keeps its historical values")
def _():
    # Anything still reading `level` must see exactly what it saw before the
    # split, or old dashboards and queries silently change meaning.
    s = EventStore(str(Path(tempfile.mkdtemp()) / "events.db"))
    for sev, cat, want in (("Critical", "intrusion", "Breach"),
                           ("High", "posture", "Posture"),
                           ("Info", "plate", "Plate"),
                           ("Critical", "weapon", "Critical"),
                           ("High", "risk", "High")):
        s.log(0, sev, 50.0, category=cat)
        assert s.query(limit=1, category=cat)[0]["level"] == want, cat


@case("site and camera identity are stamped on every event")
def _():
    s = EventStore(str(Path(tempfile.mkdtemp()) / "events.db"),
                   site={"site_id": "BOP-17", "post_name": "Kalimpong"})
    s.set_cameras({0: {"name": "Main Gate", "lat": 26.91, "lon": 88.42}})
    s.log(0, "High", 61.0)
    r = s.query(limit=1)[0]
    assert r["site_id"] == "BOP-17" and r["post_name"] == "Kalimpong"
    assert r["cam_name"] == "Main Gate" and r["lat"] == 26.91
    assert r["ts_utc"].endswith("+00:00")     # explicit zone, not a naive float
    assert r["schema_version"] >= 2


@case("event ids are unique and usable as a forward cursor")
def _():
    s = EventStore(str(Path(tempfile.mkdtemp()) / "events.db"))
    ids = [s.log(0, "High", 1.0) for _ in range(25)]
    assert len(set(ids)) == 25
    assert all(len(i) == 26 for i in ids)
    # Paging with since_id must cover every event exactly once — that is what
    # makes the C2 feed resumable after a dropped connection.
    seen, cursor = [], None
    while True:
        page = s.query(limit=7, since_id=cursor) if cursor else s.query(
            limit=7, since_id=ids[0])
        if not page:
            break
        seen += [r["event_id"] for r in page]
        cursor = page[-1]["event_id"]
    assert seen == ids[1:], "cursor skipped or repeated events"


@case("acknowledgement persists, names the operator, and cannot be overwritten")
def _():
    s = EventStore(str(Path(tempfile.mkdtemp()) / "events.db"))
    eid = s.log(0, "Critical", 96.0, category="weapon")
    assert s.query(limit=1)[0]["acked_at"] is None
    row = s.ack(eid, "Hav. R. Singh", "verified")
    assert row["acked_by"] == "Hav. R. Singh" and row["acked_at"] > 0
    s.set_ack_hash(eid, "deadbeef")
    # A second ack must not rewrite the first — chain of custody means the
    # original acknowledgement stands.
    again = s.ack(eid, "Someone Else", "")
    assert again["acked_by"] == "Hav. R. Singh"
    assert s.query(limit=1)[0]["ack_hash"] == "deadbeef"


@case("acking an unknown event returns None rather than raising")
def _():
    s = EventStore(str(Path(tempfile.mkdtemp()) / "events.db"))
    assert s.ack("NOSUCHEVENT", "op") is None


@case("MIGRATION: a schema-1 database keeps its history and gains the new columns")
def _():
    import sqlite3
    p = str(Path(tempfile.mkdtemp()) / "old.db")
    c = sqlite3.connect(p)
    c.execute(EventStore.DDL)
    for ts, lvl in ((1000.0, "Breach"), (1001.0, "Posture"),
                    (1002.0, "Critical"), (1003.0, "High")):
        c.execute("INSERT INTO events (ts, cam_id, level, score, details)"
                  " VALUES (?,?,?,?,'{}')", (ts, 0, lvl, 50.0))
    c.commit(); c.close()

    s = EventStore(p)                      # opening migrates in place
    rows = {r["level"]: r for r in s.query()}
    assert len(rows) == 4, "history was lost"
    assert (rows["Breach"]["severity"], rows["Breach"]["category"]) == ("Critical", "intrusion")
    assert (rows["Posture"]["severity"], rows["Posture"]["category"]) == ("High", "posture")
    assert (rows["Critical"]["severity"], rows["Critical"]["category"]) == ("Critical", "risk")
    assert (rows["High"]["severity"], rows["High"]["category"]) == ("High", "risk")
    for r in rows.values():
        assert r["event_id"] and r["ts_utc"], "backfill left a null hole"
    # Migration must be idempotent — the server reopens this file every start.
    s.close()
    s2 = EventStore(p)
    assert len(s2.query()) == 4


@case("prune bounds the queue but never drops undelivered events")
def _():
    s = EventStore(str(Path(tempfile.mkdtemp()) / "events.db"))
    old = s.log(0, "High", 1.0)
    s.mark_synced(old)
    kept = s.log(0, "High", 1.0)           # still unsynced
    import sqlite3
    s._conn.execute("UPDATE events SET ts = 0")   # age everything past retention
    s._conn.commit()
    s.prune(max_age_days=1.0)
    ids = [r["event_id"] for r in s.query()]
    assert old not in ids, "a delivered, expired event should be pruned"
    assert kept in ids, "an undelivered event must survive pruning"


def run() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  -- {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
