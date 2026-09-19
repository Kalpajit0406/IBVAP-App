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
def _engine(fences, cooldown=8, min_track_passes=1, **gf) -> GeoFenceEngine:
    # min_track_passes is 1 here so the geometry cases stay one-pass and
    # readable. The maturity gate that suppresses single-frame false positives
    # is exercised by its own cases at the end of this file.
    eng = GeoFenceEngine({"geofence": {"enabled": True,
                                       "reentry_cooldown_passes": cooldown,
                                       "min_track_passes": min_track_passes,
                                       **gf}})
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
    assert eng._prev_ground == {} and eng._inside == {}
    assert len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)]))) == 1   # fires anew


# ── border behaviours: dwell, counting, arming, maturity ────────────────────

@case("a single-frame detection no longer raises a breach on its own")
def _():
    # One spurious person box whose ground point lands inside a polygon used to
    # fire Critical immediately. That is the false alarm that gets a system
    # switched off, so a track must persist before it counts.
    eng = _engine([BOX], min_track_passes=3)
    assert eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)])) == []
    assert eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)])) == []
    assert len(eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)]))) == 1


@case("loitering fires once after the dwell threshold, not every pass")
def _():
    import time as _t
    fence = dict(BOX, loiter_after_s=0.05)
    eng = _engine([fence])
    b = eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)]))
    assert len(b) == 1 and b[0].event == "breach"
    assert eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)])) == []   # inside, too soon
    _t.sleep(0.06)
    out = eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)]))
    assert len(out) == 1 and out[0].event == "loiter"
    assert out[0].elapsed_s >= 0.05
    # and only once — a loiter alert that repeats every pass is a siren
    assert eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)])) == []


@case("dwelling() reports who is inside a zone and for how long")
def _():
    eng = _engine([BOX])
    eng.evaluate(_sr(0, [_det_at(1, 0.5, 0.5)]))
    d = eng.dwelling()
    assert len(d) == 1 and d[0]["track_id"] == 1 and d[0]["elapsed_s"] >= 0


@case("crossings are counted per fence and per direction")
def _():
    eng = _engine([VLINE], cooldown=1)
    # cross one way, then back
    eng.evaluate(_sr(0, [_det_at(1, 0.2, 0.5)]))
    eng.evaluate(_sr(0, [_det_at(1, 0.8, 0.5)]))
    eng.evaluate(_sr(0, [_det_at(1, 0.2, 0.5)]))
    counts = eng.crossing_counts()["by_camera"]["0"][VLINE["id"]]
    assert sum(counts.values()) == 2, counts
    assert len(counts) == 2, "the two directions should be tallied separately"


@case("an operator-labelled fence counts crossings as inbound / outbound")
def _():
    # There is no camera calibration, so 'into the country' is whatever the
    # person who drew the line said it was. The label is what makes the tally
    # mean anything in a report.
    eng = _engine([dict(VLINE, inbound="a2b")], cooldown=1)
    eng.evaluate(_sr(0, [_det_at(1, 0.2, 0.5)]))
    eng.evaluate(_sr(0, [_det_at(1, 0.8, 0.5)]))
    eng.evaluate(_sr(0, [_det_at(1, 0.2, 0.5)]))
    counts = eng.crossing_counts()["by_camera"]["0"][VLINE["id"]]
    assert set(counts) == {"inbound", "outbound"}, counts


@case("a fence outside its arming window stays quiet")
def _():
    from ibvap.geofence import Fence
    f = Fence(id="x", cam_id=0, kind="polygon", points=[(0, 0)],
              armed_from="22:00", armed_to="05:00")
    import time as _t
    midnight = _t.mktime(_t.struct_time((2026, 1, 1, 0, 30, 0, 0, 1, -1)))
    noon = _t.mktime(_t.struct_time((2026, 1, 1, 12, 0, 0, 0, 1, -1)))
    assert f.armed_at(midnight) is True, "should be armed inside a window that crosses midnight"
    assert f.armed_at(noon) is False, "should be disarmed by day"
    # An unset window means always armed — the existing default must not change.
    assert Fence(id="y", cam_id=0, kind="polygon", points=[]).armed_at(noon) is True


@case("per-fence severity survives a round trip through the store format")
def _():
    from ibvap.geofence import _parse_fence
    f = _parse_fence(dict(BOX, severity="Medium", loiter_after_s=30.0,
                          armed_from="18:00", armed_to="06:00", inbound="b2a"))
    assert f.severity == "Medium" and f.loiter_after_s == 30.0
    assert f.armed_from == "18:00" and f.inbound == "b2a"
    assert f.to_public()["severity"] == "Medium"
    # Fences written before these fields existed must still load.
    old = _parse_fence(BOX)
    assert old.severity == "Critical" and old.loiter_after_s == 0.0


# ── close-following crossing (the vision-only form of tailgating) ───────────

def _follow_events(eng, plan, cam=0):
    """plan: [(track, gx, cls), ...] applied one pass at a time; returns the
    close_following breaches that fired."""
    out = []
    for step in plan:
        dets = [_det_at(tid, gx, 0.5, cls=cls) for tid, gx, cls in step]
        out += [b for b in eng.evaluate(_sr(cam, dets)) if b.event == "close_following"]
    return out


@case("a second person crossing right behind the first is close-following")
def _():
    eng = _engine([dict(VLINE, follow_window_s=5.0)], cooldown=1)
    plan = [[(1, 0.2, "person"), (2, 0.2, "person")],
            [(1, 0.8, "person"), (2, 0.2, "person")],       # 1 crosses
            [(1, 0.8, "person"), (2, 0.8, "person")]]       # 2 crosses, right after
    ev = _follow_events(eng, plan)
    assert len(ev) == 1
    assert ev[0].track_id == 2 and ev[0].leader_track == 1
    assert 0.0 <= ev[0].elapsed_s < 5.0
    assert ev[0].event == "close_following" and ev[0].direction == "a2b"


@case("the first crosser is not flagged, and the ordinary crossings still fire")
def _():
    eng = _engine([dict(VLINE, follow_window_s=5.0)], cooldown=1)
    crossings = []
    for step in [[(1, 0.2), (2, 0.2)], [(1, 0.8), (2, 0.2)], [(1, 0.8), (2, 0.8)]]:
        crossings += eng.evaluate(_sr(0, [_det_at(t, x, 0.5) for t, x in step]))
    kinds = [(b.track_id, b.event) for b in crossings]
    assert (1, "breach") in kinds and (2, "breach") in kinds
    assert (1, "close_following") not in kinds
    assert (2, "close_following") in kinds


@case("close-following is off unless the fence sets a window")
def _():
    eng = _engine([VLINE], cooldown=1)
    plan = [[(1, 0.2, "person"), (2, 0.2, "person")],
            [(1, 0.8, "person"), (2, 0.2, "person")],
            [(1, 0.8, "person"), (2, 0.8, "person")]]
    assert _follow_events(eng, plan) == []


@case("two crossings further apart than the window are not following")
def _():
    import time as _t
    eng = _engine([dict(VLINE, follow_window_s=0.05)], cooldown=1)
    plan = [[(1, 0.2, "person"), (2, 0.2, "person")],
            [(1, 0.8, "person"), (2, 0.2, "person")]]
    assert _follow_events(eng, plan) == []
    _t.sleep(0.12)
    assert _follow_events(eng, [[(1, 0.8, "person"), (2, 0.8, "person")]]) == []


@case("the same person crossing back and forth is not following themselves")
def _():
    eng = _engine([dict(VLINE, follow_window_s=5.0)], cooldown=1)
    plan = [[(1, 0.2, "person")], [(1, 0.8, "person")],
            [(1, 0.2, "person")], [(1, 0.8, "person")]]
    assert _follow_events(eng, plan) == []


@case("opposite directions are two people passing, not one following another")
def _():
    eng = _engine([dict(VLINE, follow_window_s=5.0)], cooldown=1)
    plan = [[(1, 0.2, "person"), (2, 0.8, "person")],
            [(1, 0.8, "person"), (2, 0.8, "person")],       # 1 crosses left->right
            [(1, 0.8, "person"), (2, 0.2, "person")]]       # 2 crosses right->left
    assert _follow_events(eng, plan) == []


@case("a person crossing behind a vehicle is not tailgating")
def _():
    eng = _engine([dict(VLINE, follow_window_s=5.0)], cooldown=1)
    plan = [[(1, 0.2, "truck"), (2, 0.2, "person")],
            [(1, 0.8, "truck"), (2, 0.2, "person")],
            [(1, 0.8, "truck"), (2, 0.8, "person")]]
    assert _follow_events(eng, plan) == []
    eng = _engine([dict(VLINE, follow_window_s=5.0)], cooldown=1)
    plan = [[(1, 0.2, "truck"), (2, 0.2, "truck")],
            [(1, 0.8, "truck"), (2, 0.2, "truck")],
            [(1, 0.8, "truck"), (2, 0.8, "truck")]]
    assert len(_follow_events(eng, plan)) == 1, "a vehicle following a vehicle should count"


@case("follow_window_s round-trips and is validated")
def _():
    from ibvap.geofence import _parse_fence
    f = _parse_fence(dict(VLINE, follow_window_s=2.5))
    assert f.follow_window_s == 2.5 and f.to_public()["follow_window_s"] == 2.5
    assert _parse_fence(VLINE).follow_window_s == 0.0          # older fences load
    assert _parse_fence(dict(VLINE, follow_window_s=None)).follow_window_s == 0.0
    assert GeoFenceEngine.validate([dict(VLINE, follow_window_s=2.5)]) == []
    assert GeoFenceEngine.validate([dict(VLINE, follow_window_s=None)]) == []
    for bad in (-1, 10 ** 6, "3", True, [2]):
        assert any("follow_window_s" in e for e in
                   GeoFenceEngine.validate([dict(VLINE, follow_window_s=bad)])), bad


def _errs(**over) -> list[str]:
    return GeoFenceEngine.validate([dict(BOX, **over)])


@case("validate accepts the behaviour fields when they are well formed")
def _():
    assert _errs(severity="Medium", loiter_after_s=30, armed_from="22:00",
                 armed_to="05:30", inbound="a2b") == []
    # A fence that predates these fields, and a client that sends them empty or
    # null, must both still be accepted — they mean "unset", not "invalid".
    assert _errs() == []
    assert _errs(severity=None, loiter_after_s=None, armed_from=None,
                 armed_to=None, inbound=None) == []
    assert _errs(armed_from="", armed_to="", inbound="") == []


@case("validate rejects an unknown severity, which would otherwise be stored as-is")
def _():
    assert any("severity" in e for e in _errs(severity="banana"))
    assert any("severity" in e for e in _errs(severity="critical"))   # case matters


@case("validate rejects a loiter time that is negative, huge, or not a number")
def _():
    for bad in (-1, 10 ** 7, "30", "abc", True, [30]):
        assert any("loiter_after_s" in e for e in _errs(loiter_after_s=bad)), bad
    assert _errs(loiter_after_s=0) == []


@case("validate rejects malformed or half-set arming times")
def _():
    for bad in ("25:00", "7:30", "12:60", "noon", "1800"):
        assert any("armed_from" in e for e in _errs(armed_from=bad, armed_to="06:00")), bad
    # A lone time is treated as "always armed" by armed_at(), so a fence meant
    # to be night-only would silently stay live all day. Refuse it outright.
    assert any("together" in e for e in _errs(armed_from="22:00"))
    assert any("together" in e for e in _errs(armed_to="06:00"))


@case("validate rejects an unknown inbound direction")
def _():
    assert any("inbound" in e for e in _errs(inbound="left"))
    assert any("inbound" in e for e in _errs(inbound="both"))


@case("a fence with JSON nulls loads with defaults instead of being dropped")
def _():
    from ibvap.geofence import _parse_fence
    f = _parse_fence(dict(BOX, severity=None, loiter_after_s=None,
                          armed_from=None, armed_to=None, inbound=None))
    assert f.severity == "Critical" and f.loiter_after_s == 0.0
    assert f.armed_from == "" and f.inbound == ""
    eng = _engine([dict(BOX, severity=None, loiter_after_s=None)])
    assert eng.count == 1, "a null field silently discarded the whole fence"


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
