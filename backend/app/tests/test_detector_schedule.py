"""
test_detector_schedule.py -- which frames the detector spends a pass on.

This is the part of Detector.process_batch that decides, tick by tick, whether a
camera's frame goes to the model. It has no model in it, so it is testable with a
stub — and it matters more than it looks, because it decides what the TRACKER
sees. A phone at ~8 fps against a detector that samples at exactly 8 Hz used to
alias: some passes landed on a fresh frame and some on the frame after next, so
the step between frames actually processed alternated between one frame and two.
A person running then jumped further than their own box width on the two-frame
steps and ByteTrack lost them — measured live, the model found a runner on every
frame while the tracker turned one runner into four track ids.

The muxer hands every camera its newest frame on each 24 Hz tick, re-feeding the
SAME array until a new one arrives; the simulation below does exactly that.

    python tests/test_detector_schedule.py
    pytest  tests/test_detector_schedule.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                   # noqa: E402

from ibvap.detector import (Detector, PipelineStats, StreamResult,   # noqa: E402
                            _CamState)

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


class _Gate:
    """Stands in for the motion gate; counts how often it is consulted."""
    last_pass_was_motion = True

    def __init__(self, motion=True):
        self.motion, self.calls = motion, 0

    def should_detect(self, frame) -> bool:
        self.calls += 1
        return self.motion


class _Sched(Detector):
    """A Detector with the model and tracker removed: it records which frames
    process_batch would have sent to inference."""

    def __init__(self, detect_every=3, gating=False, motion=True):
        self.stats = PipelineStats()
        self._tick = 0
        self._cams = {}
        self._detect_every = detect_every
        self._motion_gating = gating
        self._motion = motion
        self._max_batch = 8
        self.inferred: list[tuple[int, int, float]] = []     # (tick, cam, ts)

    def _cam(self, cam_id):
        st = self._cams.get(cam_id)
        if st is None:
            st = self._cams[cam_id] = _CamState(
                tracker=None, gate=_Gate(self._motion))
        return st

    def _infer_batch(self, chunk):
        for cam, ts, frame in chunk:
            self.inferred.append((self._tick, cam, ts))
            # What Detector._track does for real: the rate limiter measures from
            # the last pass that actually ran the model.
            self._cam(cam).last_detect_tick = self._tick
        return [StreamResult(cam, ts, frame, [], 0, 0, inferred=True)
                for cam, ts, frame in chunk]

    def _carry_forward(self, cam_id, ts, frame):
        return StreamResult(cam_id, ts, frame, [], 0, 0, inferred=False, gated=True)


def _simulate(arrivals: list[float], n_ticks: int, **kw):
    """arrivals: the tick (float) at which each successive frame ARRIVES at the
    server. Each tick the muxer feeds the newest frame that has arrived, stamped
    with its arrival tick, exactly as server._muxer_loop does."""
    det = _Sched(**kw)
    frames = [np.zeros((2, 2, 3), np.uint8) for _ in arrivals]
    for tick in range(n_ticks):
        newest = None
        for i, a in enumerate(arrivals):
            if a <= tick:
                newest = i
        if newest is None:
            continue
        det.process_batch([(0, arrivals[newest], frames[newest])])
    idx = {a: i for i, a in enumerate(arrivals)}
    return det, [idx[ts] for _, _, ts in det.inferred]


def _phone(fps: float, jitter_ticks: float, n_ticks: int, seed=1) -> list[float]:
    rng, out, t = random.Random(seed), [], 0.0
    period = 24.0 / fps
    while t < n_ticks:
        out.append(max(0.0, t + rng.uniform(-jitter_ticks, jitter_ticks)))
        t += period
    return sorted(out)


@case("every frame from a phone at ~8 fps is processed exactly once, with no gaps")
def _():
    # The aliasing bug: with jitter, a fixed 8 Hz sampling skipped frames.
    for seed in range(8):
        arr = _phone(8.0, 0.9, 240, seed)
        det, idx = _simulate(arr, 240)
        assert idx == sorted(set(idx)), "a frame was inferred twice"
        assert idx == list(range(idx[0], idx[0] + len(idx))), \
            f"seed {seed}: a frame was skipped between passes: {idx}"
        assert len(idx) >= len(arr) - 3, (len(idx), len(arr))


@case("the step between processed frames never exceeds one frame for a slow source")
def _():
    arr = _phone(8.0, 0.9, 480, seed=3)
    _, idx = _simulate(arr, 480)
    steps = {b - a for a, b in zip(idx, idx[1:])}
    assert steps == {1}, steps


@case("under heavy arrival jitter far fewer frames are skipped than a fixed 8 Hz sampling would")
def _():
    # The smooth-jitter cases above are too gentle to tell the two apart. Live
    # phone arrivals are bursty (the event loop decodes other cameras' frames in
    # between), and there a fixed 8 Hz limiter measurably aliases against the
    # frames: measured when this rule was written, it skipped 8.3% of frames at
    # +/-1.5 ticks of jitter and 12.4% at +/-2.0, against 2.9% and 6.2% for
    # processing every fresh frame. What remains is frames the muxer itself
    # replaces within a single tick — not something the detector can recover.
    def skipped(jitter):
        total = missed = 0
        for seed in range(12):
            arr = _phone(8.0, jitter, 480, seed)
            _, idx = _simulate(arr, 480)
            span = idx[-1] - idx[0] + 1
            total += span
            missed += span - len(set(idx))
        return 100.0 * missed / total
    assert skipped(1.5) < 5.0, skipped(1.5)
    assert skipped(2.0) < 9.0, skipped(2.0)


@case("a very slow source (4 fps) also has every frame processed")
def _():
    arr = _phone(4.0, 1.5, 240, seed=5)
    _, idx = _simulate(arr, 240)
    assert idx == list(range(idx[0], idx[0] + len(idx)))


@case("a fast source (25 fps CCTV) is still thinned to the detection rate")
def _():
    arr = [i * 24.0 / 25.0 for i in range(int(240 * 25 / 24))]
    det, idx = _simulate(arr, 240)
    # ~one pass per 3 ticks, not one per frame — the GPU budget is unchanged.
    assert 240 / 3 * 0.85 <= len(det.inferred) <= 240 / 3 * 1.15, len(det.inferred)
    assert len(det.inferred) < len(arr) * 0.5


@case("a repeated frame never reaches inference or the motion gate")
def _():
    arr = _phone(8.0, 0.5, 240, seed=2)
    det, idx = _simulate(arr, 240, gating=True)
    gate = det._cams[0].gate
    fresh = len(set(idx))
    assert gate.calls == fresh, (gate.calls, fresh)
    assert det.stats.frames_duplicate > 0, "the repeats should have been counted"


@case("a pass that falls due on a repeat waits for the next fresh frame, not a full interval")
def _():
    # Fast source (a fresh frame every tick; ticks count from 1). First pass at
    # tick 1, so the next is due at tick 4 — but the frame handed in then is the
    # one already processed. The pass must NOT burn the slot and wait a full
    # interval (tick 7); it should run on the next fresh frame, at tick 5.
    det = _Sched()
    a, b, c, d, e = (np.zeros((2, 2, 3), np.uint8) for _ in range(5))
    feed = [a, b, c, a, d, e]                     # tick 4 repeats frame `a`
    for i, f in enumerate(feed):
        det.process_batch([(0, float(i), f)])
    ticks = [t for t, _, _ in det.inferred]
    assert ticks[:2] == [1, 5], ticks
    assert det.stats.frames_duplicate == 1


@case("the very first frames use the normal limiter until the cadence is known")
def _():
    det = _Sched()
    f = [np.zeros((2, 2, 3), np.uint8) for _ in range(3)]
    for i, fr in enumerate(f):
        det.process_batch([(0, float(i), fr)])
    assert len(det.inferred) >= 1


@case("cameras are independent: a slow phone and a fast camera in one batch")
def _():
    det = _Sched()
    phone = [np.zeros((2, 2, 3), np.uint8) for _ in range(40)]
    cctv = [np.zeros((2, 2, 3), np.uint8) for _ in range(240)]
    for tick in range(120):
        items = [(1, float(tick), cctv[tick]),                          # new frame every tick
                 (2, float(tick // 3 * 3), phone[tick // 3])]           # new frame every 3rd tick
        det.process_batch(items)
    n = lambda cam: sum(1 for _, c, _ in det.inferred if c == cam)
    assert n(2) >= 38, n(2)                       # phone: nearly every frame
    assert n(1) <= 45, n(1)                       # cctv: thinned to ~1 in 3


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
