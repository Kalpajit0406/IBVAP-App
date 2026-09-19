"""
test_uplink_tuner.py -- the adaptive uplink control law: demotion, promotion,
hysteresis, the shared bitrate budget, and (most importantly) that a congested
link settles instead of oscillating.

No network and no clock dependence: every call takes an explicit `now`.

    python tests/test_uplink_tuner.py   # prints PASS/FAIL, exits 1 on failure
    pytest  tests/test_uplink_tuner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.uplink_tuner import UplinkTuner                          # noqa: E402

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def _t(**over) -> UplinkTuner:
    cfg = {"adaptive": True, "start_rung": 2, "demote_ms": 900, "panic_ms": 2500,
           "promote_ms": 450, "promote_dwell_s": 10.0, "demote_dwell_s": 2.0,
           "uplink_budget_mbps": 12.0, "resend_s": 5.0}
    cfg.update(over)
    return UplinkTuner(cfg)


def _settle(t, cam_id, latency, now, steps=6, fps=None, step_s=1.0):
    """Feed `steps` evaluations at a fixed latency, returning the final time."""
    for _ in range(steps):
        r = t.rung(cam_id)
        t.observe(cam_id, latency, fps if fps is not None else r.fps, True, now)
        t.decide(now)
        now += step_s
    return now


@case("cold start is the configured conservative rung")
def _():
    t = _t()
    t.observe(0, 100, 8, True, now=0.0)
    assert t.rung(0).label == "960x540@8"


@case("the ladder holds a tracking-safe frame rate on every rung but the last resort")
def _():
    # Measured through the real detector: below ~6 fps ByteTrack loses a person
    # who is running (one runner became 11 track ids at 4 fps, 1 at 8 fps), and
    # a sprinter would then not register as crossing a tripwire. So only the
    # survival rung may go below it; every other step trades resolution/quality.
    t = _t()
    assert all(r.fps >= 8 for r in t.rungs[1:]), [r.label for r in t.rungs]
    assert t.rungs[0].fps >= 4
    widths = [r.w for r in t.rungs]
    assert widths == sorted(widths), "rungs must be ordered worst -> best"
    for lo, hi in zip(t.rungs, t.rungs[1:]):
        assert (hi.w, hi.fps) >= (lo.w, lo.fps)
        assert hi.mbps > lo.mbps, "a better rung must cost more bandwidth"
    assert t.rungs[-1].w >= t.rungs[0].w * 2
    # The cold start is a rung a tracker can follow a runner on.
    assert t.rungs[t.start_rung].fps >= 8


@case("sustained high latency demotes one rung")
def _():
    t = _t()
    start = t.rung(0).label
    _settle(t, 0, latency=1200, now=0.0, steps=5)
    assert t.rung(0).label != start
    assert t.status()["cameras"]["0"]["demotes"] >= 1


@case("a single high reading does not demote — hysteresis needs two")
def _():
    t = _t()
    start = t.rung(0).label
    t.observe(0, 1200, 4, True, now=0.0)
    t.decide(0.0)
    assert t.rung(0).label == start


@case("panic latency drops two rungs at once")
def _():
    t = _t(start_rung=4)
    _settle(t, 0, latency=6000, now=0.0, steps=2, step_s=3.0)
    assert t.rung(0).label == "960x540@8", t.rung(0).label


@case("latency in the dead band neither demotes nor promotes")
def _():
    t = _t()
    start = t.rung(0).label
    now = _settle(t, 0, latency=600, now=0.0, steps=40)   # between 450 and 900
    assert t.rung(0).label == start
    st = t.status()["cameras"]["0"]
    assert st["demotes"] == 0 and st["promotes"] == 0


@case("a clean link promotes, but only after the dwell time")
def _():
    t = _t()
    start_rung = t.rung(0)
    t.observe(0, 100, start_rung.fps, True, now=0.0)
    t.decide(0.0)
    t.observe(0, 100, start_rung.fps, True, now=5.0)
    t.decide(5.0)
    assert t.rung(0).label == start_rung.label, "promoted before the dwell elapsed"
    _settle(t, 0, latency=100, now=6.0, steps=10)
    assert t.rung(0).label != start_rung.label


@case("a client ignoring its rung is never promoted")
def _():
    # delivered_fps well under the rung it was told to use means the phone is
    # not obeying (an old cached page, say). Promoting it would only make the
    # shared link worse.
    t = _t()
    start = t.rung(0).label
    _settle(t, 0, latency=100, now=0.0, steps=40, fps=0.5)
    assert t.rung(0).label == start


@case("an untrusted clock never steers the rung in either direction")
def _():
    t = _t()
    start = t.rung(0).label
    for i in range(40):
        t.observe(0, 9999, 4, clock_ok=False, now=float(i))
        t.decide(float(i))
    assert t.rung(0).label == start


@case("cameras demote one at a time, worst first — never in lockstep")
def _():
    # All cameras share one bottleneck and so see much the same latency.
    # Demoting every one of them on the same evaluation overshoots badly, and
    # then they all promote back together — the synchronised-backoff
    # oscillation this rule exists to prevent.
    t = _t()
    now, first_demote = 0.0, []
    prev = {0: t.rung(0).label, 1: t.rung(1).label}
    for _ in range(12):
        t.observe(0, 1000, 4, True, now)
        t.observe(1, 1800, 4, True, now)     # clearly the worse one
        t.decide(now)
        moved = [c for c in (0, 1) if t.rung(c).label != prev[c]]
        assert len(moved) <= 1, f"{len(moved)} cameras demoted on one evaluation"
        first_demote.extend(moved)
        prev = {c: t.rung(c).label for c in (0, 1)}
        now += 1.0
    assert first_demote, "nothing demoted on a clearly congested link"
    assert first_demote[0] == 1, "the worse camera should have been cut first"


@case("the shared budget stops everyone climbing to the top rung at once")
def _():
    t = _t(uplink_budget_mbps=6.0, promote_dwell_s=1.0)
    now = 0.0
    for _ in range(120):
        for cam in range(4):
            t.observe(cam, 80, t.rung(cam).fps, True, now)
        t.decide(now)
        now += 1.0
    assert t.status()["committed_mbps"] <= 6.0 + 1e-6, t.status()["committed_mbps"]


@case("a congested link settles on a rung instead of oscillating")
def _():
    # Model a link that can carry ~1.7 Mbps: latency is fine at or below that
    # rung and terrible above it. The controller must find it and stay put.
    t = _t()
    capacity = 1.7
    now, seen = 0.0, []
    for _ in range(400):
        r = t.rung(0)
        latency = 120 if r.mbps <= capacity else 400 + (r.mbps - capacity) * 1800
        t.observe(0, latency, r.fps, True, now)
        t.decide(now)
        seen.append(t.rung(0).label)
        now += 1.0
    tail = seen[-60:]
    assert len(set(tail)) == 1, f"still oscillating: {sorted(set(tail))}"
    assert t.rung(0).mbps <= capacity, "settled above what the link can carry"


@case("tune messages are re-asserted periodically so a missed one self-heals")
def _():
    t = _t(resend_s=5.0)
    t.observe(0, 100, 4, True, now=0.0)
    assert 0 in t.decide(0.0)                 # first contact
    assert t.decide(1.0) == {}                # nothing new to say
    t.observe(0, 100, 4, True, now=7.0)
    assert 0 in t.decide(7.0), "should have re-asserted after resend_s"


@case("a disconnected camera stops counting against the budget")
def _():
    t = _t()
    t.observe(0, 100, 4, True, now=0.0)
    t.observe(1, 100, 4, True, now=0.0)
    t.decide(0.0)
    before = t.status()["committed_mbps"]
    t.forget(1)
    assert t.status()["committed_mbps"] < before
    assert "1" not in t.status()["cameras"]


@case("disabled tuner emits nothing and leaves the phone on its defaults")
def _():
    t = _t(adaptive=False)
    t.observe(0, 9999, 1, True, now=0.0)
    assert t.decide(0.0) == {}


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


if __name__ == "__main__":
    sys.exit(run())
