"""
test_ws_capture.py -- capture-age frame dropping and the clock trust that gates
it. No network, no camera: frames are built in-memory in the browser's wire
format ([8-byte LE capture-ms | JPEG]) and pushed straight into the capture.

The behaviour under test is safety-critical in both directions. Not dropping
stale frames is what let a congested hotspot link accumulate 10-15 s of latency
at a live demo. But dropping on a *wrong* clock would discard every frame and
black the camera out entirely, which is worse. So age-dropping must engage on a
genuinely late frame and must disengage when the clock stops making sense.

    python tests/test_ws_capture.py     # prints PASS/FAIL, exits 1 on failure
    pytest  tests/test_ws_capture.py
"""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402

from ibvap import ws_capture                                        # noqa: E402
from ibvap.ws_capture import WebSocketCapture                       # noqa: E402

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


_JPEG = cv2.imencode(".jpg", np.zeros((720, 1280, 3), np.uint8))[1].tobytes()


def _wire(cap_ms: int) -> bytes:
    """One frame in the browser's wire format."""
    return struct.pack("<Q", int(cap_ms)) + _JPEG


def _cap(offset_ms: float = 0.0) -> WebSocketCapture:
    """A capture whose clock is already synced to ours."""
    c = WebSocketCapture(0)
    c._clock_offset_ms = offset_ms
    c._have_offset = True
    return c


def _now_ms() -> float:
    return time.time() * 1000.0


@case("a fresh frame is decoded and queued")
def _():
    ws_capture.set_max_frame_age_ms(1200)
    c = _cap()
    c.push_frame(_wire(_now_ms()))
    assert c.frames_received == 1
    assert c.frames_stale == 0


@case("a frame older than max_frame_age_ms is dropped without decoding")
def _():
    ws_capture.set_max_frame_age_ms(1200)
    c = _cap()
    c.push_frame(_wire(_now_ms() - 12_000))        # the demo's 12 s backlog
    assert c.frames_stale == 1
    assert c.frames_received == 0                   # never reached the queue
    # The camera must still read as live — it is sending, just far behind — or
    # the UI would show it as disconnected instead of delayed.
    assert c.connected is False or c._last_frame_at > 0


@case("a dropped frame still updates the latency figure the operator sees")
def _():
    ws_capture.set_max_frame_age_ms(1200)
    c = _cap()
    c.push_frame(_wire(_now_ms() - 9_000))
    assert c.frames_stale == 1
    assert c.latency_ms > 5000, c.latency_ms      # the 9 s is reported, not hidden


@case("max_frame_age_ms = 0 disables age-dropping entirely")
def _():
    ws_capture.set_max_frame_age_ms(0)
    try:
        c = _cap()
        c.push_frame(_wire(_now_ms() - 30_000))
        assert c.frames_stale == 0 and c.frames_received == 1
    finally:
        ws_capture.set_max_frame_age_ms(1200)


@case("a phone with a wildly wrong clock is never trusted, so nothing is dropped")
def _():
    ws_capture.set_max_frame_age_ms(1200)
    # Clock off by an hour: every reading is absurd, clock_ok stays False, and
    # the frames must keep flowing rather than being silently discarded.
    c = _cap(offset_ms=-3_600_000)
    for _ in range(10):
        c.push_frame(_wire(_now_ms()))
    assert c.clock_ok is False
    assert c.frames_stale == 0
    assert c.frames_received == 10


@case("sane readings earn clock trust")
def _():
    ws_capture.set_max_frame_age_ms(1200)
    c = _cap()
    for _ in range(10):
        c.push_frame(_wire(_now_ms() - 120))       # a believable 120 ms delay
    assert c.clock_ok is True
    assert c.frames_stale == 0


@case("dropping nearly everything backs off instead of blinding the camera")
def _():
    # A clock skewed by ~5 s reads as 'sane' (inside the plausible band) but
    # makes every frame look stale. Without the auto-disable the camera would
    # go permanently black. It must resume delivering frames instead.
    ws_capture.set_max_frame_age_ms(1200)
    c = _cap(offset_ms=-5_000)
    for _ in range(200):
        c.push_frame(_wire(_now_ms()))
    assert c.clock_ok is True                      # readings look plausible
    assert c.frames_stale > 0                      # it did drop for a while
    assert c.frames_received > 0, "camera was blinded by a skewed clock"


@case("a bare JPEG with no timestamp is still accepted")
def _():
    ws_capture.set_max_frame_age_ms(1200)
    c = _cap()
    c.push_frame(_JPEG)
    assert c.frames_received == 1 and c.frames_stale == 0


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
