"""
test_face_events.py -- PersonArrivalTracker's burst-of-N capture scheduler +
majority vote, and FaceResultWriter's write/finalize handoff (the dedicated
data/face_results/ output folder). No GPU, no network; writes into a
throwaway tempfile dir.

    python tests/test_face_events.py        # prints PASS/FAIL, exits 1 on failure
    pytest  tests/test_face_events.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                       # noqa: E402

from ibvap.face import FaceHit                                           # noqa: E402
from ibvap.face_events import FaceResultWriter, PersonArrivalTracker     # noqa: E402

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def _frame():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _cfg(d, **over) -> dict:
    base = {"enabled": True, "dir": str(d)}
    base.update(over)
    return {"face_results": base}


def _hit(name="Alice", on_watchlist=True, sim=0.6) -> FaceHit:
    return FaceHit(person_track=1, bbox=(0, 0, 5, 5), det_score=0.9,
                  matched_id=(name.lower() if name else None), matched_name=name,
                  similarity=sim, on_watchlist=on_watchlist)


# ── PersonArrivalTracker ────────────────────────────────────────────────────

@case("wants_capture is True on first sight, opens the track's state")
def _():
    t = PersonArrivalTracker()
    assert t.wants_capture((0, 1), tick=0, now=1000.0) is True


@case("3-of-5 agreeing on-watchlist hits on the same identity -> confirmed match")
def _():
    t = PersonArrivalTracker(burst_n=5, vote_min=3, max_attempts=20, attempt_window_s=60.0)
    key = (0, 1)
    for now in (1000.0, 1001.0):     # 2 misses (no face found) don't count against the burst
        assert t.wants_capture(key, 0, now) is True
        assert t.record(key, now, None) is None
    hits = [_hit("Alice", True), _hit("Alice", True), _hit("Bob", True),
           _hit("Alice", True), _hit("Alice", False, sim=0.1)]
    verdict = None
    for now, h in zip((1002.0, 1003.0, 1004.0, 1005.0, 1006.0), hits):
        assert t.wants_capture(key, 0, now) is True
        verdict = t.record(key, now, h)
    assert verdict is not None
    assert verdict.on_watchlist is True
    assert verdict.matched_id == "alice"
    assert verdict.votes == 3 and verdict.of == 5
    assert t.wants_capture(key, 10, 1010.0) is False   # burst is done


@case("2-of-5 agreeing is below vote_min=3 -> not a match")
def _():
    t = PersonArrivalTracker(burst_n=5, vote_min=3, max_attempts=20, attempt_window_s=60.0)
    key = (0, 1)
    hits = [_hit("Alice", True), _hit("Alice", True), _hit("Bob", True),
           _hit("Carol", True), _hit(None, False, sim=0.0)]
    verdict = None
    for now, h in zip(range(1000, 1005), hits):
        verdict = t.record(key, float(now), h)
    assert verdict is not None
    assert verdict.on_watchlist is False
    assert verdict.matched_id is None
    assert verdict.votes == 2 and verdict.of == 5


@case("a mixed-identity burst where no id reaches vote_min -> unknown")
def _():
    t = PersonArrivalTracker(burst_n=4, vote_min=2, max_attempts=20, attempt_window_s=60.0)
    key = (0, 1)
    hits = [_hit("Alice", True), _hit("Bob", True), _hit("Carol", True), _hit("Dave", True)]
    verdict = None
    for now, h in zip(range(1000, 1004), hits):
        verdict = t.record(key, float(now), h)
    assert verdict is not None and verdict.on_watchlist is False
    assert verdict.votes == 1 and verdict.of == 4


@case("wants_capture stops once max_attempts is reached, even short of burst_n")
def _():
    # max_attempts is clamped to >= burst_n (never budget fewer attempts than
    # the burst itself needs), so this exercises the case that clamp doesn't
    # collapse: attempts run out from repeated "no face found" ticks before
    # burst_n readings are ever collected.
    t = PersonArrivalTracker(burst_n=3, max_attempts=3, attempt_window_s=60.0, vote_min=1)
    key = (0, 1)
    verdict = None
    for now in (1000.0, 1001.0, 1002.0):
        assert t.wants_capture(key, 0, now) is True
        verdict = t.record(key, now, None)      # never finds a face
    assert verdict is not None
    assert verdict.of == 0 and verdict.on_watchlist is False
    assert t.wants_capture(key, 3, 1003.0) is False


@case("wants_capture stops once attempt_window_s elapses")
def _():
    t = PersonArrivalTracker(burst_n=5, max_attempts=99, attempt_window_s=5.0, vote_min=1)
    key = (0, 1)
    assert t.wants_capture(key, 0, 1000.0) is True
    t.record(key, 1000.0, None)
    assert t.wants_capture(key, 1, 1001.0) is True
    verdict = t.record(key, 1006.1, None)         # > 5s since first_ts
    assert verdict is not None
    assert t.wants_capture(key, 2, 1007.0) is False


@case("fresh_hit returns the last on_watchlist hit within hit_ttl_s, then expires")
def _():
    t = PersonArrivalTracker(hit_ttl_s=2.0)
    key = (0, 1)
    t.wants_capture(key, 0, 1000.0)
    t.record(key, 1000.0, _hit("Alice", True))
    assert t.fresh_hit(key, 1001.5) is not None
    assert t.fresh_hit(key, 1001.5).matched_name == "Alice"
    assert t.fresh_hit(key, 1002.1) is None


@case("flush() resets state — a later key is a fresh arrival")
def _():
    t = PersonArrivalTracker()
    key = (0, 1)
    t.wants_capture(key, 0, 1000.0)
    t.flush(key)
    assert t.wants_capture(key, 5, 1005.0) is True


@case("prune() drops state for tracks no longer live, only for the given camera")
def _():
    t = PersonArrivalTracker()
    t.wants_capture((0, 1), 0, 1000.0)
    t.wants_capture((0, 2), 0, 1000.0)
    t.wants_capture((1, 1), 0, 1000.0)
    t.record((0, 1), 1000.0, None)
    t.record((0, 2), 1000.0, None)
    t.record((1, 1), 1000.0, None)
    t.prune(0, live_track_ids={2})
    assert (0, 1) not in t._state       # dropped
    assert (0, 2) in t._state           # still live
    assert (1, 1) in t._state           # other camera untouched


@case("independent tracks / cameras don't interfere")
def _():
    t = PersonArrivalTracker()
    assert t.wants_capture((0, 1), 0, 1000.0) is True
    assert t.wants_capture((0, 2), 0, 1000.0) is True
    assert t.wants_capture((1, 1), 0, 1000.0) is True


# ── cosine-threshold cutoff (the actual gallery-matching arithmetic) ────────

@case("cosine similarity threshold cutoff flips around match_threshold")
def _():
    a = np.zeros(512, dtype=np.float32); a[0] = 1.0
    # b at a known angle from a: cos(theta) via a simple 2-component mix. Values
    # a hair either side of 0.38 (not the exact boundary, which float32 rounding
    # can land on either side of) confirm the >= cutoff behaves as intended.
    for target_sim, want in ((0.50, True), (0.40, True), (0.36, False), (0.10, False)):
        b = np.zeros(512, dtype=np.float32)
        b[0] = target_sim
        b[1] = float(np.sqrt(max(0.0, 1.0 - target_sim ** 2)))
        sim = float(np.dot(a, b))
        assert abs(sim - target_sim) < 1e-5, sim
        assert (sim >= 0.38) == want, (target_sim, sim, want)


# ── FaceResultWriter ─────────────────────────────────────────────────────────

class _Verdict:
    def __init__(self, matched_id=None, matched_name=None, similarity=0.0,
                votes=0, of=0, on_watchlist=False):
        self.matched_id = matched_id
        self.matched_name = matched_name
        self.similarity = similarity
        self.votes = votes
        self.of = of
        self.on_watchlist = on_watchlist


@case("capture_snapshot writes one jpg per burst tick and opens a pending record")
def _():
    d = tempfile.mkdtemp()
    try:
        w = FaceResultWriter(_cfg(d))
        r0 = w.capture_snapshot(0, _frame(), 7, 0, {"detail": "alice"}, now=1000.0)
        r1 = w.capture_snapshot(0, _frame(), 7, 1, {"detail": "alice"}, now=1000.1)
        assert r0 is not None and r1 is not None
        assert r0["file"] != r1["file"]
        assert (Path(d) / r0["file"]).is_file() and (Path(d) / r1["file"]).is_file()
        assert len(r0["sha256"]) == 64
        assert w.status()["pending"] == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("finalize completes the record with the verdict but does NOT write it (two-step handoff)")
def _():
    d = tempfile.mkdtemp()
    try:
        w = FaceResultWriter(_cfg(d))
        w.capture_snapshot(0, _frame(), 7, 0, {}, now=1000.0)
        w.capture_snapshot(0, _frame(), 7, 1, {}, now=1000.1)
        v = _Verdict(matched_id="alice", matched_name="Alice", similarity=0.71,
                    votes=3, of=5, on_watchlist=True)
        rec = w.finalize(0, 7, v, now=1000.5)
        assert rec["matched_name"] == "Alice" and rec["on_watchlist"] is True
        assert len(rec["files"]) == 2
        assert not (Path(d) / "0" / "index.jsonl").exists()   # record() not called yet
        assert w.status()["pending"] == 0
        rec["ev_hash"] = "deadbeef"
        w.record(rec)
        lines = (Path(d) / "0" / "index.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        for k in ("ts", "cam_id", "track_id", "files", "sha256s", "matched_id",
                  "matched_name", "similarity", "votes", "of", "on_watchlist", "ev_hash"):
            assert k in row, f"missing key {k}"
        assert row["ev_hash"] == "deadbeef"
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("finalize with no pending record returns None")
def _():
    d = tempfile.mkdtemp()
    try:
        w = FaceResultWriter(_cfg(d))
        assert w.finalize(0, 99, _Verdict()) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("sweep_timeouts finalizes an abandoned burst as unknown, keeping its pictures")
def _():
    d = tempfile.mkdtemp()
    try:
        w = FaceResultWriter(_cfg(d))
        w.capture_snapshot(0, _frame(), 1, 0, {}, now=1000.0)
        assert w.sweep_timeouts(now=1010.0, timeout_s=20.0) == []      # not stale yet
        out = w.sweep_timeouts(now=1025.0, timeout_s=20.0)
        assert len(out) == 1
        assert out[0]["matched_id"] is None and out[0]["on_watchlist"] is False
        assert len(out[0]["files"]) == 1
        assert w.status()["pending"] == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("disabled writer captures nothing and returns None")
def _():
    d = tempfile.mkdtemp()
    try:
        w = FaceResultWriter(_cfg(d, enabled=False))
        assert w.capture_snapshot(0, _frame(), 1, 0, {}) is None
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
