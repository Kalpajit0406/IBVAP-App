"""
test_geofence.py -- GeoFenceEngine geometry + breach logic. No model, no GPU,
no disk: hand-built Detection / StreamResult and fence dicts.

    python tests/test_geofence.py        # prints PASS/FAIL, exits 1 on failure
    pytest  tests/test_geofence.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.detector import Detection, StreamResult              # noqa: E402
from ibvap.geofence import (GeoFenceEngine, ground_point,        # noqa: E402
                          point_in_polygon, segments_intersect)


# ── builders ────────────────────────────────────────────────────────────────
def _engine(fences, cooldown=8) -> GeoFenceEngine:
    eng = GeoFenceEngine({"geofence": {"enabled": True,
                                       "reentry_cooldown_passes": cooldown}})
    eng.set_fences(fences)
    return eng


def _det(tid: int, bbox, cls: str = "person") -> Detection:
    return Detection(track_id=tid, class_id=0 if cls == "person" else 2,
                     class_name=cls, bbox=bbox, confidence=0.9,
                     is_person=(cls == "person"), is_vehicle=(cls != "person"))


def _det_at(tid: int, gx: float, gy: float, w=1280, h=720, cls="person") -> Detection:
    """A detection whose GROUND POINT (bbox bottom-centre) sits at (gx, gy) norm."""
    cx, by = gx * w, gy * h
    return _det(tid, (cx - 12, by - 48, cx + 12, by), cls)


def _sr(cam: int, dets, shape=(720, 1280, 3)) -> StreamResult:
    frame = types.SimpleNamespace(shape=shape)
    return StreamResult(cam_id=cam, timestamp=0.0, frame=frame, detections=list(dets),
                        person_count=sum(d.is_person for d in dets),
                        vehicle_count=sum(d.is_vehicle for d in dets))


BOX = {"cam_id": 0, "kind": "polygon", "id": "poly1",
       "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
       "targets": ["any"], "enabled": True}
VLINE = {"cam_id": 0, "kind": "line", "id": "line1",
         "points": [[0.5, 0.1], [0.5, 0.9]],
         "direction": "both", "targets": ["any"], "enabled": True}


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# ── geometry helpers ────────────────────────────────────────────────────────
@case("ground_point is bbox bottom-centre, normalised")
def _():
    assert ground_point((100, 100, 300, 500), 1000, 1000) == (0.2, 0.5)


@case("point_in_polygon: inside vs outside a square")
def _():
    sq = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    assert point_in_polygon((0.5, 0.5), sq) is True
    assert point_in_polygon((0.05, 0.5), sq) is False


@case("segments_intersect: crossing vs same-side")
def _():
    assert segments_intersect((0.2, 0.5), (0.8, 0.5), (0.5, 0.1), (0.5, 0.9)) is True
    assert segments_intersect((0.2, 0.5), (0.3, 0.5), (0.5, 0.1), (0.5, 0.9)) is False


# ── polygon breaches ────────────────────────────────────────────────────────
@case("target inside a polygon -> one breach; outside -> none")
def _():
    eng = _engine([BOX])
    assert len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)]))) == 1
    eng2 = _engine([BOX])
    assert eng2.evaluate(_sr(0, [_det_at(1, 0.05, 0.05)])) == []


@case("polygon breach is edge-triggered: 5 passes inside -> 1 breach")
def _():
    eng = _engine([BOX])
    fired = sum(len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)]))) for _ in range(5))
    assert fired == 1, fired


@case("leave the polygon (>= exit_passes) then re-enter -> 2 breaches")
def _():
    eng = _engine([BOX])
    n = len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)])))          # 1
    for _ in range(3):                                            # sustained exit
        n += len(eng.evaluate(_sr(0, [_det_at(1, 0.05, 0.05)])))
    n += len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)])))        # 1
    assert n == 2, n


@case("a single outside blip does NOT end a polygon breach (exit debounce)")
def _():
    eng = _engine([BOX])                       # exit_passes defaults to 2
    n = len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)])))         # 1 (enter)
    n += len(eng.evaluate(_sr(0, [_det_at(1, 0.05, 0.5)])))       # one pass outside
    n += len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)])))        # back in — no new breach
    assert n == 1, n
    assert eng.active_zones(0) == {"poly1": True}


@case("class filter: a person can't breach a vehicle-only fence")
def _():
    veh_only = {**BOX, "targets": ["vehicle"]}
    eng = _engine([veh_only])
    assert eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5, cls="person")])) == []
    eng2 = _engine([veh_only])
    assert len(eng2.evaluate(_sr(0, [_det_at(1, 0.5, 0.5, cls="truck")]))) == 1


@case("disabled fence never breaches")
def _():
    eng = _engine([{**BOX, "enabled": False}])
    assert eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)])) == []


@case("breach outcome is resolution-independent (720p vs 1080p)")
def _():
    def run(w, h):
        eng = _engine([BOX])
        b = eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5, w, h)], shape=(h, w, 3)))
        return b
    a, c = run(1280, 720), run(1920, 1080)
    assert len(a) == len(c) == 1
    assert abs(a[0].ground_point[0] - c[0].ground_point[0]) < 1e-3
    assert abs(a[0].ground_point[1] - c[0].ground_point[1]) < 1e-3


# ── tripwire lines ──────────────────────────────────────────────────────────
@case("line: prev->curr crossing fires; staying one side does not")
def _():
    eng = _engine([VLINE])
    assert eng.evaluate(_sr(0, [_det_at(1, 0.2, 0.5)])) == []      # sets prev, no cross
    assert len(eng.evaluate(_sr(0, [_det_at(1, 0.8, 0.5)]))) == 1  # crosses
    eng2 = _engine([VLINE])
    eng2.evaluate(_sr(0, [_det_at(1, 0.2, 0.5)]))
    assert eng2.evaluate(_sr(0, [_det_at(1, 0.3, 0.5)])) == []     # same side


@case("line direction filter: wrong-way crossing is ignored")
def _():
    a2b = {**VLINE, "direction": "a2b"}
    eng = _engine([a2b])
    eng.evaluate(_sr(0, [_det_at(1, 0.2, 0.5)]))
    b = eng.evaluate(_sr(0, [_det_at(1, 0.8, 0.5)]))              # left->right == a2b
    assert len(b) == 1 and b[0].direction == "a2b", b
    eng2 = _engine([a2b])
    eng2.evaluate(_sr(0, [_det_at(1, 0.8, 0.5)]))
    assert eng2.evaluate(_sr(0, [_det_at(1, 0.2, 0.5)])) == []    # right->left == b2a, filtered


@case("tripwire re-fire is debounced, then fires again after the cooldown")
def _():
    eng = _engine([VLINE], cooldown=3)
    seq = []
    for gx in (0.2, 0.8, 0.2, 0.8, 0.2):        # establish, cross, x4 more
        seq.append(len(eng.evaluate(_sr(0, [_det_at(1, gx, 0.5)]))))
    assert seq == [0, 1, 0, 0, 1], seq


@case("polyline line: crossing ANY segment of a bent fence fires once")
def _():
    L = {"cam_id": 0, "kind": "line", "id": "poly_l", "direction": "both",
         "points": [[0.2, 0.2], [0.2, 0.8], [0.8, 0.8]], "targets": ["any"],
         "enabled": True}
    eng = _engine([L])                          # cross the 2nd (horizontal) segment
    eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.6)]))
    assert len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.95)]))) == 1
    eng2 = _engine([L])                         # cross the 1st (vertical) segment
    eng2.evaluate(_sr(0, [_det_at(1, 0.1, 0.5)]))
    assert len(eng2.evaluate(_sr(0, [_det_at(1, 0.35, 0.5)]))) == 1
    eng3 = _engine([L])                         # a move that crosses neither
    eng3.evaluate(_sr(0, [_det_at(1, 0.5, 0.2)]))
    assert eng3.evaluate(_sr(0, [_det_at(1, 0.6, 0.4)])) == []


@case("polyline validate: line accepts 2+ points, rejects 1")
def _():
    assert GeoFenceEngine.validate(
        [{"cam_id": 0, "kind": "line", "points": [[0.1, 0.1], [0.5, 0.5], [0.9, 0.2]]}]) == []
    errs = GeoFenceEngine.validate([{"cam_id": 0, "kind": "line", "points": [[0.1, 0.1]]}])
    assert errs and "at least 2" in errs[0], errs


# ── engine bookkeeping ──────────────────────────────────────────────────────
@case("active_zones reports exactly the fence being breached")
def _():
    left = {"cam_id": 0, "kind": "polygon", "id": "L",
            "points": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], "enabled": True}
    right = {"cam_id": 0, "kind": "polygon", "id": "R",
             "points": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]], "enabled": True}
    eng = _engine([left, right])
    eng.evaluate(_sr(0, [_det_at(1, 0.25, 0.5)]))
    az = eng.active_zones(0)
    assert az == {"L": True, "R": False}, az


@case("prune clears stale track state; a returning track fires again")
def _():
    eng = _engine([BOX])
    assert len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)]))) == 1
    eng.prune(0, set())                              # track 1 gone
    assert eng._prev_ground == {} and eng._inside == set()
    assert len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)]))) == 1   # fires anew


@case("validate rejects malformed fences")
def _():
    bad = [
        {"cam_id": 0, "kind": "polygon", "points": [[0.1, 0.1], [0.2, 0.2]]},   # <3
        {"cam_id": 0, "kind": "line", "points": [[0, 0]]},                      # <2
        {"cam_id": 0, "kind": "polygon", "points": [[1.4, 0.2], [0, 0], [1, 1]]},  # oob
        {"cam_id": 0, "kind": "blob", "points": [[0, 0], [1, 0], [1, 1]]},      # kind
        {"cam_id": 0, "kind": "line", "points": [[0, 0], [1, 1]], "direction": "sideways"},
        {"kind": "line", "points": [[0, 0], [1, 1]]},                           # no cam_id
    ]
    errs = GeoFenceEngine.validate(bad)
    assert len(errs) >= 6, errs
    # a 3+-point line is now a valid polyline fence
    assert GeoFenceEngine.validate([BOX, VLINE]) == []
    assert GeoFenceEngine.validate(
        [{"cam_id": 0, "kind": "line", "points": [[0, 0], [0.5, 0.5], [1, 0]]}]) == []


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
