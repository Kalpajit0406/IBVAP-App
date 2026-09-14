"""
test_learn.py -- Harvester candidate logic + label format. No GPU, no model:
hand-built Detection lists, a temp pool dir, blank frames.

    python tests/test_learn.py
    pytest  tests/test_learn.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.detector import Detection                          # noqa: E402
from ibvap.learn import Harvester, to_yolo_line               # noqa: E402

FRAME = np.zeros((720, 1280, 3), np.uint8)


def _cfg(pool: Path, **over) -> dict:
    h = {"min_conf": 0.55, "min_track_frames": 3, "ambiguous_conf": 0.25,
         "per_cam_rate_limit_s": 2.0, "max_pool": 100, "jpeg_quality": 70,
         "classes": [0, 2, 3, 5, 7]}
    h.update(over)
    return {"learning": {"enabled": True, "paths": {"pool": str(pool)}, "harvest": h}}


def _det(tid: int, cls: int = 0, conf: float = 0.8, bbox=(500, 300, 560, 600)) -> Detection:
    return Detection(track_id=tid, class_id=cls,
                     class_name="person" if cls == 0 else "car", bbox=bbox,
                     confidence=conf, is_person=(cls == 0), is_vehicle=(cls != 0))


def _fresh() -> tuple[Harvester, Path]:
    d = Path(tempfile.mkdtemp())
    return Harvester(_cfg(d)), d


def _drain(h: Harvester, want: int, timeout: float = 3.0) -> int:
    end = time.time() + timeout
    while time.time() < end and h.status()["harvested_total"] < want:
        time.sleep(0.05)
    return h.status()["harvested_total"]


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("to_yolo_line: normalised, 6dp, class id unchanged")
def _():
    ln = to_yolo_line(2, (100, 100, 300, 500), 1000, 1000)
    assert ln == "2 0.200000 0.300000 0.200000 0.400000\n", repr(ln)


@case("track must be stable (>= min_track_frames inferred passes) to harvest")
def _():
    h, _d = _fresh()
    for _ in range(2):                       # min_track_frames is 3
        h.submit(0, FRAME, [_det(1)], inferred=True, ts=time.time())
    time.sleep(0.3)
    assert h.status()["harvested_total"] == 0, "harvested before stable"
    h.submit(0, FRAME, [_det(1)], inferred=True, ts=time.time())
    assert _drain(h, 1) == 1
    h.stop()


@case("a carried (inferred=False) pass does not advance track stability")
def _():
    h, _d = _fresh()
    for _ in range(5):
        h.submit(0, FRAME, [_det(1)], inferred=False, ts=time.time())
    time.sleep(0.3)
    assert h.status()["harvested_total"] == 0
    h.stop()


@case("per-camera rate limit: two fast stable submits -> one row")
def _():
    h, _d = _fresh()
    for _ in range(4):
        h.submit(0, FRAME, [_det(1)], inferred=True, ts=time.time())
    assert _drain(h, 1) == 1
    h.submit(0, FRAME + 1, [_det(1)], inferred=True, ts=time.time())   # within 2s
    time.sleep(0.3)
    assert h.status()["harvested_total"] == 1, "rate limit not enforced"
    h.stop()


@case("clean-frame gate: an ambiguous-confidence box skips the whole frame")
def _():
    h, _d = _fresh()
    for _ in range(4):
        h.submit(0, FRAME,
                 [_det(1, conf=0.8), _det(2, cls=2, conf=0.4)],   # 0.4 in [0.25, 0.55)
                 inferred=True, ts=time.time())
    time.sleep(0.3)
    assert h.status()["harvested_total"] == 0
    assert h.status()["skipped_ambiguous"] >= 1
    h.stop()


@case("dedup: the identical frame submitted repeatedly -> one row")
def _():
    h, d = _fresh()
    for _ in range(4):
        h.submit(0, FRAME, [_det(1)], inferred=True, ts=time.time())
    assert _drain(h, 1) == 1
    # advance past the rate limit but keep the exact same pixels
    h._last_grab[0] = 0.0
    h.submit(0, FRAME, [_det(1)], inferred=True, ts=time.time())
    time.sleep(0.3)
    assert h.status()["harvested_total"] == 1, "identical frame harvested twice"
    h.stop()


@case("max_pool: harvesting stops once the pending pool is full")
def _():
    d = Path(tempfile.mkdtemp())
    h = Harvester(_cfg(d, max_pool=2, per_cam_rate_limit_s=0.0))
    for i in range(6):
        h._last_sig.clear()
        for _ in range(4):
            h.submit(0, FRAME + i, [_det(1)], inferred=True, ts=time.time())
        time.sleep(0.15)
    assert h.status()["harvested_total"] <= 3, h.status()
    h.stop()


@case("apply_review updates verdict counts; validate_review rejects bad input")
def _():
    h, d = _fresh()
    for _ in range(4):
        h.submit(0, FRAME, [_det(1)], inferred=True, ts=time.time())
    assert _drain(h, 1) == 1
    hid = next(iter(h._index))
    (d / "verdicts.jsonl").write_text(
        '{"id": "%s", "verdict": "keep", "boxes": null, "ts": 0}\n' % hid)
    h.apply_review(hid, "keep", None)
    assert h.status()["pool_kept"] == 1 and h.status()["pool_pending"] == 0
    assert Harvester.validate_review({"id": hid, "verdict": "maybe"}), "bad verdict accepted"
    assert Harvester.validate_review({"id": "../etc", "verdict": "keep"}), "path id accepted"
    assert Harvester.validate_review({"id": hid, "verdict": "keep"}) == []
    h.stop()


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
