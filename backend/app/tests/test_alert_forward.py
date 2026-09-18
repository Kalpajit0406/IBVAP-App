"""
test_alert_forward.py -- outbound alert delivery: severity filtering, ordering,
store-and-forward across an outage, and per-sink isolation.

This is the path that turns "an alert was raised" into "someone else knows".
Until it existed, EventStore's `synced` column and `unsynced()`/`mark_synced()`
were dead code and every event sat undelivered forever, so the behaviour worth
pinning down is mostly about failure: what happens when the link is down, and
what happens when it comes back.

No network — sinks are replaced with a recording fake.

    python tests/test_alert_forward.py
    pytest  tests/test_alert_forward.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.alert_forward import AlertForwarder                     # noqa: E402
from ibvap.event_store import EventStore                           # noqa: E402
from ibvap.sinks import Sink, event_payload, severity_at_least     # noqa: E402

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


class FakeSink(Sink):
    """Records what it was given; can be told to fail."""
    name = "fake"

    def __init__(self, min_severity="Info", label="fake", down=False):
        super().__init__({"enabled": True, "min_severity": min_severity,
                          "label": label})
        self.got: list[str] = []
        self.down = down
        self.attempts = 0

    def send(self, row):
        self.attempts += 1
        if self.down:
            return False, "simulated outage"
        self.got.append(row["event_id"])
        return True, "ok"


def _fwd(sinks, **cfg) -> tuple[AlertForwarder, EventStore]:
    store = EventStore(str(Path(tempfile.mkdtemp()) / "events.db"))
    f = AlertForwarder(store, {"enabled": True, "backoff_base_s": 0.0, **cfg})
    f._sinks = [type(f._sinks[0])(s) if f._sinks else None for s in []] or []
    from ibvap.alert_forward import _SinkState
    f._sinks = [_SinkState(s) for s in sinks]
    return f, store


@case("an event is delivered to an enabled sink and then marked synced")
def _():
    s = FakeSink()
    f, store = _fwd([s])
    eid = store.log(0, "Critical", 96.0, category="weapon")
    f._drain_once()
    assert s.got == [eid]
    assert store.unsynced_count() == 0


@case("a sink only receives events at or above its minimum severity")
def _():
    low, high = FakeSink(label="all"), FakeSink("Critical", label="critical-only")
    f, store = _fwd([low, high])
    store.log(0, "Info", 10.0, category="plate")
    crit = store.log(0, "Critical", 96.0, category="weapon")
    f._drain_once()
    assert len(low.got) == 2
    assert high.got == [crit], "a filtered event must not reach this sink"
    # Filtering is not failure: the event is still fully delivered.
    assert store.unsynced_count() == 0


@case("events are delivered in order")
def _():
    s = FakeSink()
    f, store = _fwd([s])
    ids = [store.log(0, "High", 50.0) for _ in range(10)]
    f._drain_once()
    assert s.got == ids


@case("STORE-AND-FORWARD: nothing is lost or delivered while the link is down")
def _():
    s = FakeSink(down=True)
    f, store = _fwd([s])
    ids = [store.log(0, "High", 50.0) for _ in range(5)]
    for _ in range(3):
        f._drain_once()
    assert s.got == []
    assert store.unsynced_count() == 5, "undelivered events must stay queued"

    # Link returns: the backlog drains, in order, exactly once.
    s.down = False
    f._sinks[0].retry_after = 0.0
    f._drain_once()
    assert s.got == ids
    assert store.unsynced_count() == 0


@case("a failing sink does not stall a healthy one")
def _():
    dead, ok = FakeSink(label="dead", down=True), FakeSink(label="ok")
    f, store = _fwd([dead, ok])
    ids = [store.log(0, "High", 50.0) for _ in range(3)]
    f._drain_once()
    assert ok.got == ids, "a healthy sink should have received everything"
    assert dead.got == []
    # The events stay queued because one sink still owes them — that is what
    # makes the outage recoverable rather than silent data loss.
    assert store.unsynced_count() == 3


@case("a recovered sink does not re-send what it already took")
def _():
    a, b = FakeSink(label="a"), FakeSink(label="b", down=True)
    f, store = _fwd([a, b])
    ids = [store.log(0, "High", 50.0) for _ in range(3)]
    f._drain_once()
    b.down = False
    f._sinks[1].retry_after = 0.0
    f._drain_once()
    assert a.got == ids, "sink a must not receive duplicates"
    assert b.got == ids


@case("backoff delays a retry instead of hammering a dead endpoint")
def _():
    s = FakeSink(down=True)
    f, store = _fwd([s], backoff_base_s=60.0)
    store.log(0, "High", 50.0)
    f._drain_once()
    first = s.attempts
    f._drain_once()
    f._drain_once()
    assert s.attempts == first, "should still be backing off, not retrying"
    assert f._sinks[0].failures >= 1


@case("max_attempts drops a poisoned event rather than blocking the queue")
def _():
    s = FakeSink(down=True)
    f, store = _fwd([s], max_attempts=2, backoff_base_s=0.0)
    bad = store.log(0, "High", 50.0)
    for _ in range(4):
        f._sinks[0].retry_after = 0.0
        f._drain_once()
    good = store.log(0, "High", 50.0)
    s.down = False
    f._sinks[0].retry_after = 0.0
    f._drain_once()
    assert good in s.got, "the queue stayed blocked behind the failed event"
    assert f._sinks[0].dropped >= 1


@case("with no sinks configured the queue still drains, so disk cannot fill")
def _():
    f, store = _fwd([])
    store.log(0, "High", 50.0)
    f._drain_once()
    assert store.unsynced_count() == 0


@case("the wire payload carries site, camera and evidence identity")
def _():
    store = EventStore(str(Path(tempfile.mkdtemp()) / "events.db"),
                       site={"site_id": "BOP-17", "post_name": "Kalimpong"})
    store.set_cameras({0: {"name": "Main Gate", "lat": 26.9, "lon": 88.4}})
    store.log(0, "Critical", 96.0, 1, 0,
              {"type": "face_match", "matched_name": "alice"}, "sha-abc",
              category="face_match")
    p = event_payload(store.query(limit=1)[0])
    assert p["schema"] == "ibvap.event/2"
    assert p["severity"] == "Critical" and p["category"] == "face_match"
    assert p["site"]["site_id"] == "BOP-17"
    assert p["camera"]["name"] == "Main Gate" and p["camera"]["lat"] == 26.9
    assert p["evidence"]["sha256"] == "sha-abc"
    assert p["details"]["matched_name"] == "alice"
    assert p["ts_utc"].endswith("+00:00")
    assert p["event_id"] and len(p["event_id"]) == 26


@case("severity ordering is total and Info is the floor")
def _():
    order = ["Info", "Low", "Medium", "High", "Critical"]
    for i, lo in enumerate(order):
        for j, hi in enumerate(order):
            assert severity_at_least(hi, lo) == (j >= i), f"{hi} >= {lo}"


@case("status reports queue depth and per-sink health for the operator")
def _():
    s = FakeSink(down=True, label="C2 webhook")
    f, store = _fwd([s])
    store.log(0, "High", 50.0)
    f._drain_once()
    st = f.status()
    assert st["pending"] == 1
    sink = st["sinks"][0]
    assert sink["label"] == "C2 webhook"
    assert sink["failures"] >= 1 and sink["last_error"]


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
