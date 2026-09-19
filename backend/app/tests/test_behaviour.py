"""
test_behaviour.py -- running and group-formation rules (ibvap/behaviour.py).

These are heuristics with no measured false-alarm rate behind them, so the cases
worth pinning down are the ways they could cry wolf: walking, standing still with
box jitter, a tracker swapping two people, one spurious detection, a queue that
flickers. No model, no GPU: hand-built detections on a synthetic clock.

    python tests/test_behaviour.py       # prints PASS/FAIL, exits 1 on failure
    pytest  tests/test_behaviour.py
"""
from __future__ import annotations

import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.behaviour import BehaviourEngine                      # noqa: E402
from ibvap.detector import Detection, StreamResult               # noqa: E402

DT = 0.125          # one detection pass at detect_fps 8
H = 100             # a person 100 px tall on a 720 px frame (0.139 of it)


def _person(tid: int, gx: float, gy: float = 500.0, h: float = H, person=True) -> Detection:
    """A detection whose GROUND POINT (bbox bottom-centre) is (gx, gy)."""
    return Detection(track_id=tid, class_id=0 if person else 2,
                     class_name="person" if person else "car",
                     bbox=(gx - 0.2 * h, gy - h, gx + 0.2 * h, gy), confidence=0.9,
                     is_person=person, is_vehicle=not person)


def _sr(t: float, dets, cam: int = 0) -> StreamResult:
    frame = types.SimpleNamespace(shape=(720, 1280, 3))
    return StreamResult(cam_id=cam, timestamp=t, frame=frame, detections=list(dets),
                        person_count=sum(d.is_person for d in dets),
                        vehicle_count=sum(d.is_vehicle for d in dets))


def _engine(**over) -> BehaviourEngine:
    cfg = {"behaviour": {"enabled": True}}
    for section in ("running", "group"):
        if section in over:
            cfg["behaviour"][section] = over.pop(section)
    cfg["behaviour"].update(over)
    return BehaviourEngine(cfg)


def _run(eng, speed_bh_s: float, secs: float, *, h=H, tid=1, start_x=20.0, t0=0.0):
    """One person moving in a straight line; returns [(t, event), ...]."""
    fired, x, t = [], start_x, t0
    for _ in range(int(secs / DT)):
        for ev in eng.evaluate(_sr(t, [_person(tid, x, h=h)])):
            fired.append((t, ev))
        x += speed_bh_s * h * DT
        t += DT
    return fired


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# ── running ─────────────────────────────────────────────────────────────────

@case("walking pace never triggers running")
def _():
    assert _run(_engine(), 0.8, 8.0) == []


@case("a sustained run fires once, about a second in, and not again while it lasts")
def _():
    fired = _run(_engine(), 2.5, 5.0)
    assert len(fired) == 1, [(round(t, 2), e.kind) for t, e in fired]
    t, ev = fired[0]
    assert 1.0 <= t <= 2.2, f"fired at {t:.2f}s"
    assert ev.kind == "running" and ev.severity == "Medium"
    assert 2.0 <= ev.details["speed_bh_s"] <= 3.0
    assert ev.details["track_id"] == 1


@case("standing still with box jitter never reads as movement")
def _():
    rng = random.Random(7)
    eng, t, fired = _engine(), 0.0, []
    for _ in range(int(20 / DT)):
        fired += eng.evaluate(_sr(t, [_person(1, 500 + rng.uniform(-4, 4),
                                              500 + rng.uniform(-4, 4),
                                              H + rng.uniform(-3, 3))]))
        t += DT
    assert fired == []


@case("a tracker jump (two people swapped) is not a sprint")
def _():
    # Someone standing still is suddenly reported 400 px away — 32 body-heights
    # per second, faster than anyone runs. Left in the speed window it would
    # read as a sprint for a whole second; it must be treated as a glitch.
    eng, t, fired = _engine(), 0.0, []
    for i in range(int(6 / DT)):
        x = 300 if t < 2.0 else 700
        fired += eng.evaluate(_sr(t, [_person(1, x)]))
        t += DT
    assert fired == []


@case("people too small in frame to measure are ignored")
def _():
    # 30 px tall = 4% of frame height: a couple of pixels of jitter is a large
    # fraction of a body-height, so their motion means nothing.
    assert _run(_engine(), 2.5, 5.0, h=30) == []


@case("a new track id every pass has no history and never fires")
def _():
    eng, t, x, fired = _engine(), 0.0, 20.0, []
    for i in range(40):
        fired += eng.evaluate(_sr(t, [_person(100 + i, x)]))
        x += 2.5 * H * DT
        t += DT
    assert fired == []


@case("vehicles are never 'running'")
def _():
    eng, t, x, fired = _engine(), 0.0, 20.0, []
    for _ in range(40):
        fired += eng.evaluate(_sr(t, [_person(1, x, person=False)]))
        x += 4 * H * DT
        t += DT
    assert fired == []


@case("slowing down re-arms the alert, but the cooldown stops a stutter")
def _():
    eng = _engine()
    f1 = _run(eng, 2.5, 3.0)                              # run
    f2 = _run(eng, 0.0, 2.0, start_x=900, t0=3.0)         # stop
    f3 = _run(eng, 2.5, 3.0, start_x=900, t0=5.0)         # run again, 5 s later
    assert len(f1) == 1 and f2 == []
    assert f3 == [], "a second alert inside the cooldown is a stream, not a report"
    eng2 = _engine(running={"cooldown_s": 0.0})
    n = len(_run(eng2, 2.5, 3.0)) + len(_run(eng2, 0.0, 2.0, start_x=900, t0=3.0)) \
        + len(_run(eng2, 2.5, 3.0, start_x=900, t0=5.0))
    assert n == 2, n


@case("the speed threshold is honoured")
def _():
    assert _run(_engine(), 1.5, 5.0) == []                 # jog < default 2.0
    assert len(_run(_engine(running={"speed_bh_s": 1.2}), 1.5, 5.0)) == 1


@case("a disabled engine or a disabled rule emits nothing")
def _():
    assert _run(_engine(enabled=False), 2.5, 4.0) == []
    assert _run(_engine(running={"enabled": False}), 2.5, 4.0) == []


@case("prune drops people who have been gone past the grace period")
def _():
    eng = _engine()
    eng.evaluate(_sr(0.0, [_person(1, 100)]))
    eng.evaluate(_sr(0.2, [_person(1, 100)]))
    assert (0, 1) in eng._tracks
    eng.prune(0, live_track_ids=set())          # gone, but only just: kept
    assert (0, 1) in eng._tracks
    eng.evaluate(_sr(20.0, [_person(2, 100)]))  # 20 s on, track 1 never came back
    eng.prune(0, live_track_ids={2})
    assert (0, 1) not in eng._tracks and (0, 2) in eng._tracks


# ── group formation ─────────────────────────────────────────────────────────

def _together(eng, xs, secs, t0=0.0, tids=None, gy=500.0):
    """People standing at ground x positions `xs`; returns [(t, event)]."""
    tids = tids or list(range(1, len(xs) + 1))
    fired, t = [], t0
    for _ in range(int(secs / DT)):
        dets = [_person(tid, x, gy) for tid, x in zip(tids, xs)]
        for ev in eng.evaluate(_sr(t, dets)):
            fired.append((t, ev))
        t += DT
    return fired


@case("three people standing together for a few seconds form one group")
def _():
    fired = _together(_engine(), [500, 560, 620], 8.0)
    assert len(fired) == 1, [(round(t, 2), e.kind) for t, e in fired]
    t, ev = fired[0]
    assert 3.0 <= t <= 4.0, f"fired at {t:.2f}s"
    assert ev.kind == "group" and ev.severity == "Medium"
    assert ev.details["size"] == 3 and ev.details["members"] == [1, 2, 3]


@case("people far apart are not a group")
def _():
    assert _together(_engine(), [100, 500, 900], 8.0) == []


@case("two people are a pair, not a group")
def _():
    assert _together(_engine(), [500, 560], 8.0) == []


@case("a gathering that breaks up before the minimum duration is ignored")
def _():
    eng = _engine()
    a = _together(eng, [500, 560, 620], 2.0)
    b = _together(eng, [100, 500, 900], 4.0, t0=2.0)
    assert a == [] and b == []


@case("a chain counts as one group even when its ends are far apart")
def _():
    # 0.9 bh between neighbours (inside the 1.5 radius) but 1.8 bh end to end.
    fired = _together(_engine(), [400, 490, 580], 6.0)
    assert len(fired) == 1 and fired[0][1].details["size"] == 3


@case("one person joining or leaving does not restart or repeat the alert")
def _():
    eng, t, fired = _engine(), 0.0, []
    for i in range(int(9 / DT)):
        tids = [1, 2, 3]
        if t >= 2.0:
            tids = [1, 2, 3, 4]          # someone joins
        if t >= 5.0:
            tids = [2, 3, 4]             # someone leaves
        dets = [_person(tid, 500 + 60 * k) for k, tid in enumerate(tids)]
        fired += eng.evaluate(_sr(t, dets))
        t += DT
    assert len(fired) == 1, len(fired)


@case("a spurious track cannot complete a group")
def _():
    # Two real people plus a third track id that is new on every pass.
    eng, t, fired = _engine(), 0.0, []
    for i in range(int(8 / DT)):
        fired += eng.evaluate(_sr(t, [_person(1, 500), _person(2, 560),
                                      _person(100 + i, 620)]))
        t += DT
    assert fired == []


@case("a group that flickers apart and re-forms is one event, not a stream")
def _():
    eng = _engine()
    a = _together(eng, [500, 560, 620], 4.5)
    apart = _together(eng, [100, 500, 900], 3.0, t0=4.5)       # past the grace
    b = _together(eng, [500, 560, 620], 5.0, t0=7.5)
    assert len(a) == 1 and apart == []
    assert b == [], "re-forming inside the cooldown must not alert again"
    eng2 = _engine(group={"cooldown_s": 0.0})
    n = len(_together(eng2, [500, 560, 620], 4.5)) \
        + len(_together(eng2, [100, 500, 900], 3.0, t0=4.5)) \
        + len(_together(eng2, [500, 560, 620], 5.0, t0=7.5))
    assert n == 2, n


@case("vehicles beside people do not count toward a group")
def _():
    eng, t, fired = _engine(), 0.0, []
    for _ in range(int(8 / DT)):
        fired += eng.evaluate(_sr(t, [_person(1, 500), _person(2, 560),
                                      _person(3, 620, person=False)]))
        t += DT
    assert fired == []


@case("the overlay marks a runner and a confirmed group, and lets go afterwards")
def _():
    eng = _engine()
    _run(eng, 2.5, 3.0)
    ov = eng.overlay(0)
    assert ov["running"] == {1}, ov
    eng.evaluate(_sr(60.0, []))                       # a minute later, nobody there
    assert eng.overlay(0)["running"] == set()

    eng = _engine()
    _together(eng, [500, 560, 620], 5.0)
    g = eng.overlay(0)["groups"]
    assert len(g) == 1 and g[0]["members"] == {1, 2, 3}
    assert eng.overlay(99) == {"running": set(), "groups": []}   # unknown camera


@case("status reports what fired")
def _():
    eng = _engine()
    _run(eng, 2.5, 3.0)
    _together(eng, [500, 560, 620], 5.0, t0=10.0, tids=[5, 6, 7])
    s = eng.status()
    assert s["running"]["events"] == 1 and s["group"]["events"] == 1


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
