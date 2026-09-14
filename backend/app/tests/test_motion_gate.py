"""
test_motion_gate.py -- MotionGate frame-difference pre-filter. No model, no GPU.

    python tests/test_motion_gate.py
    pytest  tests/test_motion_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.motion_gate import MotionGate                     # noqa: E402

H, W = 720, 1280


def _blank(v: int = 40) -> np.ndarray:
    return np.full((H, W, 3), v, dtype=np.uint8)


def _with_block(v: int = 40, box=(200, 200, 500, 500), c: int = 220) -> np.ndarray:
    f = _blank(v)
    x1, y1, x2, y2 = box
    f[y1:y2, x1:x2] = c
    return f


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("first frame always passes (no baseline yet)")
def _(_g):
    g = MotionGate(force_every=1000, hold_frames=0)
    assert g.should_detect(_blank()) is True


@case("a static scene is skipped once the keep-alive window is disabled")
def _(_g):
    g = MotionGate(force_every=1000, hold_frames=0)
    g.should_detect(_blank())                       # baseline
    skipped = [not g.should_detect(_blank()) for _ in range(20)]
    assert all(skipped), "static frames should all be skipped"
    assert g.stats.skipped >= 20


@case("a moving object opens the gate")
def _(_g):
    g = MotionGate(force_every=1000, hold_frames=0)
    g.should_detect(_blank())
    assert g.should_detect(_with_block()) is True
    assert g.last_pass_was_motion is True


@case("force_every sweeps a static scene on schedule")
def _(_g):
    g = MotionGate(force_every=5, hold_frames=0)
    g.should_detect(_blank())                       # baseline (counts as a pass)
    results = [g.should_detect(_blank()) for _ in range(10)]
    # every 5th call is forced through even with zero motion
    assert results[4] is True and results[9] is True
    assert results[0] is False and results[3] is False
    assert g.stats.forced >= 2


@case("hold_frames keeps the gate open briefly after motion stops")
def _(_g):
    g = MotionGate(force_every=1000, hold_frames=3)
    g.should_detect(_blank())                                   # baseline
    g.should_detect(_with_block(box=(200, 200, 500, 500)))      # object appears
    g.should_detect(_with_block(box=(400, 400, 700, 700)))      # object moves -> hold armed
    still = _with_block(box=(400, 400, 700, 700))               # object now stationary
    tail = [g.should_detect(still) for _ in range(3)]           # 3 hold frames
    assert tail == [True, True, True], tail
    assert g.should_detect(still) is False                      # hold exhausted -> skip


@case("skip_pct reflects the skipped ratio")
def _(_g):
    g = MotionGate(force_every=1000, hold_frames=0)
    g.should_detect(_blank())
    for _ in range(9):
        g.should_detect(_blank())
    assert 80.0 <= g.stats.skip_pct <= 100.0, g.stats.skip_pct


def run() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn(None)
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
