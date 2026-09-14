"""
test_snapshots.py -- SnapshotWriter (per-camera folders, annotated + raw JPEG,
index.jsonl, debounce) and the malicious_postures trigger filter. No GPU, no
network; writes into a throwaway tempfile dir.

    python tests/test_snapshots.py        # prints PASS/FAIL, exits 1 on failure
    pytest  tests/test_snapshots.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402

from ibvap.snapshots import SnapshotWriter, malicious_postures  # noqa: E402


# ── builders ────────────────────────────────────────────────────────────────
def _cfg(d, **over) -> dict:
    base = {"enabled": True, "dir": str(d), "cooldown_s": 5.0}
    base.update(over)
    return {"snapshots": base}


def _frame():
    return np.zeros((64, 96, 3), dtype=np.uint8)


def _person(lying=False, crouch=False, aim=False, is_person=True, tid=1):
    pf = types.SimpleNamespace(lying=lying, crouching=crouch, chest_aim=aim,
                               arms_up=False, vigorous=False)
    return types.SimpleNamespace(is_person=is_person, posture=pf, track_id=tid)


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# ── capture ─────────────────────────────────────────────────────────────────
@case("capture writes annotated + raw jpg under <dir>/<cam>/ with 64-hex shas")
def _():
    d = tempfile.mkdtemp()
    try:
        w = SnapshotWriter(_cfg(d))
        r = w.capture(0, _frame(), _frame(), "breach",
                      {"detail": "north gate", "track_id": 7})
        assert r is not None
        assert r["file"] == "0/" + Path(r["file"]).name
        assert (Path(d) / r["file"]).is_file()
        assert (Path(d) / r["raw_file"]).is_file()
        assert Path(r["raw_file"]).name.endswith("_raw.jpg")
        assert "north-gate" in r["file"]
        assert len(r["sha256"]) == 64 and len(r["raw_sha256"]) == 64
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("capture with no annotated frame still writes the raw jpg")
def _():
    d = tempfile.mkdtemp()
    try:
        w = SnapshotWriter(_cfg(d))
        r = w.capture(0, None, _frame(), "weapon", {"track_id": 2})
        assert r is not None and r["file"] is None and r["sha256"] is None
        assert r["raw_file"] is not None and len(r["raw_sha256"]) == 64
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("record appends exactly one index.jsonl line with the documented keys")
def _():
    d = tempfile.mkdtemp()
    try:
        w = SnapshotWriter(_cfg(d))
        r = w.capture(1, _frame(), _frame(), "posture",
                      {"detail": "CROUCH", "track_id": 3})
        r["ev_hash"] = "abc123"
        w.record(r)
        lines = (Path(d) / "1" / "index.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        for k in ("ts", "reason", "cam_id", "track_id", "detail", "file",
                  "raw_file", "sha256", "raw_sha256", "deduped", "ev_hash"):
            assert k in row, f"missing key {k}"
        assert row["reason"] == "posture" and row["cam_id"] == 1
        assert row["ev_hash"] == "abc123"
        assert w.status()["by_reason"]["posture"] == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("identical frame in one pass is written once, reused for later records")
def _():
    d = tempfile.mkdtemp()
    try:
        w = SnapshotWriter(_cfg(d, cooldown_s=0.0))
        f = _frame()                                    # same pixels both calls
        r1 = w.capture(0, f, f, "breach", {"track_id": 1})
        r2 = w.capture(0, f, f, "breach", {"track_id": 2})
        assert r1["deduped"] is False and r2["deduped"] is True
        assert r2["file"] == r1["file"] and r2["sha256"] == r1["sha256"]
        jpgs = sorted(p.name for p in (Path(d) / "0").glob("*.jpg"))
        assert len(jpgs) == 2                            # one annotated + one raw, not four
        f2 = f.copy(); f2[0, 0] = 255                    # different frame → new files
        r3 = w.capture(0, f2, f2, "breach", {"track_id": 3})
        assert r3["deduped"] is False and r3["file"] != r1["file"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("recent() returns the tail newest-first")
def _():
    d = tempfile.mkdtemp()
    try:
        w = SnapshotWriter(_cfg(d, cooldown_s=0.0))
        for i in range(3):
            r = w.capture(0, _frame(), _frame(), "breach", {"track_id": i})
            r["ev_hash"] = f"h{i}"
            w.record(r)
        rows = w.recent(0, limit=2)
        assert [r["track_id"] for r in rows] == [2, 1]
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── debounce ────────────────────────────────────────────────────────────────
@case("should_capture debounces per (cam, track, reason) within cooldown_s")
def _():
    d = tempfile.mkdtemp()
    try:
        w = SnapshotWriter(_cfg(d, cooldown_s=10.0))
        t0 = 1000.0
        assert w.should_capture(0, 5, "breach", now=t0) is True
        assert w.should_capture(0, 5, "breach", now=t0 + 3) is False
        assert w.should_capture(0, 6, "breach", now=t0 + 3) is True    # other track
        assert w.should_capture(0, 5, "weapon", now=t0 + 3) is True    # other reason
        assert w.should_capture(0, 5, "breach", now=t0 + 11) is True   # cooldown elapsed
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("prune() keeps a recent guard through a flicker, clears it once cooled")
def _():
    d = tempfile.mkdtemp()
    try:
        w = SnapshotWriter(_cfg(d, cooldown_s=100.0))
        assert w.should_capture(0, 5, "breach", now=1000.0) is True
        w.prune(0, {6, 7}, now=1050.0)                                 # gone, still cooling
        assert w.should_capture(0, 5, "breach", now=1051.0) is False   # guard held
        w.prune(0, {6, 7}, now=1000.0 + 200)                           # gone AND cooled
        assert w.should_capture(0, 5, "breach", now=1201.0) is True    # state cleared
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("on_* flags gate should_capture by reason")
def _():
    d = tempfile.mkdtemp()
    try:
        w = SnapshotWriter(_cfg(d, on_posture=False))
        assert w.should_capture(0, 1, "posture") is False
        assert w.should_capture(0, 1, "breach") is True
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── folders / disabled ─────────────────────────────────────────────────────
@case("ensure_cam creates the per-camera folder")
def _():
    d = tempfile.mkdtemp()
    try:
        w = SnapshotWriter(_cfg(d))
        w.ensure_cam(3)
        assert (Path(d) / "3").is_dir()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case("disabled writer captures nothing and returns None")
def _():
    d = tempfile.mkdtemp()
    try:
        w = SnapshotWriter(_cfg(d, enabled=False))
        assert w.capture(0, _frame(), _frame(), "breach", {}) is None
        assert w.should_capture(0, 1, "breach") is False
        assert list(Path(d).iterdir()) == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── malicious_postures ─────────────────────────────────────────────────────
@case("malicious_postures picks lying/crouching persons, skips aim/standing/non-person")
def _():
    dets = [_person(lying=True, tid=1), _person(crouch=True, tid=2),
            _person(aim=True, tid=3), _person(tid=4),
            _person(lying=True, is_person=False, tid=5),
            types.SimpleNamespace(is_person=True, posture=None, track_id=9)]
    out = malicious_postures(dets)
    assert [d.track_id for d in out] == [1, 2]
    assert all(d.posture.lying or d.posture.crouching for d in out)
    assert malicious_postures([]) == []


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
