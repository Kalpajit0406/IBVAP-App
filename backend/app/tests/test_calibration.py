"""
test_calibration.py -- Calibrator EMA / suppression / bounds. Pure numpy.

    python tests/test_calibration.py
    pytest  tests/test_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.calibration import Calibrator                      # noqa: E402


def _cfg(**over) -> dict:
    c = {"enabled": True, "ema_alpha": 0.3, "decay_per_day": 0.5,
         "headroom": 0.05, "max_tighten": 0.30, "grid": [32, 18],
         "promote_cell_after": 5, "min_obs_before_apply": 10}
    c.update(over)
    return {"learning": {"calibration": c}}


class _D:
    def __init__(self, tid, bbox, conf=0.8, cls=0):
        self.track_id = tid; self.bbox = bbox; self.confidence = conf; self.class_id = cls


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("floors are untouched until min_obs_before_apply observations")
def _():
    cal = Calibrator(_cfg(min_obs_before_apply=50))
    for _ in range(20):
        cal.observe(0, [_D(1, (10, 10, 20, 20))], set())   # tid vanishes -> obs++
    assert cal.person_floor(0, 0.30) == 0.30
    assert cal.vehicle_floor(0, 0.40) == 0.40


@case("repeated confident-but-transient FPs raise the person floor, clamped to max_tighten")
def _():
    cal = Calibrator(_cfg())
    for _ in range(200):
        # a track that appears at high conf and dies immediately (1 pass)
        cal.observe(0, [_D(9999 + _ % 7, (400, 300, 460, 600), conf=0.9)], set())
    d = cal.person_floor(0, 0.30) - 0.30
    assert 0.0 < d <= 0.30 + 1e-6, d


@case("a delta never exceeds [-headroom, +max_tighten]")
def _():
    cal = Calibrator(_cfg())
    for _ in range(500):
        cal.observe(0, [_D(1234 + _ % 3, (0, 0, 30, 30), conf=0.99)], set())
    st = cal._cams[0]
    assert -0.05 - 1e-6 <= st.person_delta <= 0.30 + 1e-6


@case("a grid cell with >= promote_cell_after died-young hits becomes a suppression cell")
def _():
    cal = Calibrator(_cfg(promote_cell_after=5, min_obs_before_apply=1))
    box = (100, 500, 140, 620)                     # ground point ~ (0.094, 0.86)
    for _ in range(6):
        cal.observe(0, [_D(1, box, conf=0.5)], set())
    gp = Calibrator.gp_norm(box, 1280, 720)
    assert cal.is_suppressed(0, gp) is True
    # a spot with no history is not suppressed
    assert cal.is_suppressed(0, (0.9, 0.1)) is False


@case("one hit short of promote_cell_after does NOT suppress")
def _():
    cal = Calibrator(_cfg(promote_cell_after=6, min_obs_before_apply=1))
    box = (600, 400, 640, 520)
    for _ in range(5):
        cal.observe(0, [_D(1, box, conf=0.5)], set())
    assert cal.is_suppressed(0, Calibrator.gp_norm(box, 1280, 720)) is False


@case("a stable track (never dies young) does not raise the floor")
def _():
    cal = Calibrator(_cfg())
    for i in range(300):
        cal.observe(0, [_D(1, (400, 300, 460, 600), conf=0.7)],
                    {1})                              # tid stays live -> passes accrue
    cal.observe(0, [], set())                          # now it vanishes with many passes -> "accepted"
    assert cal.person_floor(0, 0.30) <= 0.31


@case("decay after +1 day halves the learned counts")
def _():
    cal = Calibrator(_cfg())
    for _ in range(40):
        cal.observe(0, [_D(1, (100, 500, 140, 620), conf=0.6)], set())
    before = float(cal._cams[0].rej.sum())
    cal._last_decay -= 86400.0
    cal.decay()
    after = float(cal._cams[0].rej.sum())
    assert 0.45 * before <= after <= 0.55 * before, (before, after)


@case("snapshot() -> restore() round-trips deltas + suppression")
def _():
    cal = Calibrator(_cfg(min_obs_before_apply=1))
    for _ in range(8):
        cal.observe(0, [_D(1, (100, 500, 140, 620), conf=0.9)], set())
    snap = cal.snapshot()
    cal2 = Calibrator(_cfg(min_obs_before_apply=1))
    cal2.restore({"cams": snap["cams"]})
    assert abs(cal2.person_floor(0, 0.30) - cal.person_floor(0, 0.30)) < 1e-3
    assert cal2._cams[0].suppress == cal._cams[0].suppress


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
