"""
test_anpr_events.py -- VehicleArrivalTracker's arrival/followup/idle state
machine, and AnprResultWriter's write/finalize handoff (the dedicated
data/anpr_results/ output folder). No GPU, no network; writes into a
throwaway tempfile dir.

    python tests/test_anpr_events.py        # prints PASS/FAIL, exits 1 on failure
    pytest  tests/test_anpr_events.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                     # noqa: E402

from ibvap.anpr_events import AnprResultWriter, VehicleArrivalTracker  # noqa: E402

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def _frame():
    return np.zeros((64, 96, 3), dtype=np.uint8)


def _cfg(d, **over) -> dict:
    base = {"enabled": True, "dir": str(d)}
    base.update(over)
    return {"anpr_results": base}


def _reading(text="WB06AB1234", conf=0.9, valid=True):
    return types.SimpleNamespace(text=text, conf=conf, valid=valid)


# ── VehicleArrivalTracker ──────────────────────────────────────────────────

@case("first sighting is 'arrival'")
def _():
    t = VehicleArrivalTracker(follow_up_max=3, follow_up_s=10.0)
    assert t.classify((0, 1), tick=0, now=1000.0, confident=False) == "arrival"


@case("subsequent ticks are 'followup' until the max, then 'done'")
def _():
    t = VehicleArrivalTracker(follow_up_max=2, follow_up_s=100.0)
    key = (0, 1)
    assert t.classify(key, 0, 1000.0, False) == "arrival"
    assert t.classify(key, 1, 1001.0, False) == "followup"
    assert t.classify(key, 2, 1002.0, False) == "followup"
    assert t.classify(key, 3, 1003.0, False) == "done"
    assert t.classify(key, 4, 1004.0, False) == "done"    # stays done


@case("confident=True short-circuits straight to 'done'")
def _():
    t = VehicleArrivalTracker(follow_up_max=8, follow_up_s=100.0)
    key = (0, 1)
    assert t.classify(key, 0, 1000.0, False) == "arrival"
    assert t.classify(key, 1, 1001.0, True) == "done"
    assert t.classify(key, 2, 1002.0, False) == "done"


@case("follow-up time window expires even under the max-reads budget")
def _():
    t = VehicleArrivalTracker(follow_up_max=99, follow_up_s=5.0)
    key = (0, 1)
    assert t.classify(key, 0, 1000.0, False) == "arrival"
    assert t.classify(key, 1, 1001.0, False) == "followup"
    assert t.classify(key, 2, 1006.1, False) == "done"     # > 5s since first_ts


@case("flush() resets state — a later key is a fresh arrival")
def _():
    t = VehicleArrivalTracker()
    key = (0, 1)
    t.classify(key, 0, 1000.0, False)
    t.flush(key)
    assert t.classify(key, 5, 1005.0, False) == "arrival"


@case("prune() drops state for tracks no longer live, only for the given camera")
def _():
    t = VehicleArrivalTracker()
    t.classify((0, 1), 0, 1000.0, False)
    t.classify((0, 2), 0, 1000.0, False)
    t.classify((1, 1), 0, 1000.0, False)
    t.prune(0, live_track_ids={2})
    assert t.classify((0, 1), 1, 1001.0, False) == "arrival"   # dropped -> fresh
    assert t.classify((0, 2), 1, 1001.0, False) == "followup"  # still live -> continues
    assert t.classify((1, 1), 1, 1001.0, False) == "followup"  # other camera untouched


@case("independent tracks / cameras don't interfere")
def _():
    t = VehicleArrivalTracker()
    assert t.classify((0, 1), 0, 1000.0, False) == "arrival"
    assert t.classify((0, 2), 0, 1000.0, False) == "arrival"
    assert t.classify((1, 1), 0, 1000.0, False) == "arrival"


# ── AnprResultWriter ────────────────────────────────────────────────────────

@case("capture_snapshot writes one jpg and opens a pending record")
def _():
    d = tempfile.mkdtemp()
    try:
        w = AnprResultWriter(_cfg(d))
        r = w.capture_snapshot(0, _frame(), 7, {"detail": "car"}, now=1000.0)
        assert r is not None
        assert r["file"] == "0/" + Path(r["file"]).name
        assert (Path(d) / r["file"]).is_file()
        assert len(r["sha256"]) == 64
        assert r["plate"] is None and r["vehicle_type"] is None
        assert w.status()["pending"] == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("attach_vehicle_type fills in the pending record")
def _():
    d = tempfile.mkdtemp()
    try:
        w = AnprResultWriter(_cfg(d))
        w.capture_snapshot(0, _frame(), 7, {}, now=1000.0)
        vt = types.SimpleNamespace(label="car", conf=0.87, fallback=False)
        w.attach_vehicle_type(0, 7, vt)
        rec = w.finalize(0, 7, _reading())
        assert rec["vehicle_type"] == "car"
        assert rec["vehicle_type_conf"] == 0.87
        assert rec["vehicle_type_fallback"] is False
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("finalize completes the record but does NOT write it (two-step handoff)")
def _():
    d = tempfile.mkdtemp()
    try:
        w = AnprResultWriter(_cfg(d))
        w.capture_snapshot(0, _frame(), 7, {}, now=1000.0)
        rec = w.finalize(0, 7, _reading(text="DL3CAB123", conf=0.75))
        assert rec["plate"] == "DL3CAB123" and rec["plate_conf"] == 0.75
        assert not (Path(d) / "0" / "index.jsonl").exists()   # record() not called yet
        assert w.status()["pending"] == 0                      # but no longer pending
        rec["ev_hash"] = "deadbeef"
        w.record(rec)
        lines = (Path(d) / "0" / "index.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        for k in ("ts", "cam_id", "track_id", "plate", "plate_conf", "plate_valid",
                  "vehicle_type", "vehicle_type_conf", "vehicle_type_fallback",
                  "file", "sha256", "ev_hash"):
            assert k in row, f"missing key {k}"
        assert row["ev_hash"] == "deadbeef"
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("finalize with no pending record returns None")
def _():
    d = tempfile.mkdtemp()
    try:
        w = AnprResultWriter(_cfg(d))
        assert w.finalize(0, 99, _reading()) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("sweep_timeouts finalizes stale pending records with plate=null")
def _():
    d = tempfile.mkdtemp()
    try:
        w = AnprResultWriter(_cfg(d, finalize_timeout_s=5.0))
        w.capture_snapshot(0, _frame(), 1, {}, now=1000.0)
        assert w.sweep_timeouts(now=1003.0) == []               # not stale yet
        out = w.sweep_timeouts(now=1006.0)
        assert len(out) == 1
        assert out[0]["plate"] is None and out[0]["track_id"] == 1
        assert w.status()["pending"] == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("disabled writer captures nothing and returns None")
def _():
    d = tempfile.mkdtemp()
    try:
        w = AnprResultWriter(_cfg(d, enabled=False))
        assert w.capture_snapshot(0, _frame(), 1, {}) is None
        assert list(Path(d).iterdir()) == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
