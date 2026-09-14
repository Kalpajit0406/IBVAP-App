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
