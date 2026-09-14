"""
test_anpr.py -- AnprEngine region-format cleanup (`_coerce_in`) and the
(cam_id, track_id) reading store. No model, no GPU, no EasyOCR: the engine is
constructed pointing at a non-existent weights file so it stays disabled
(`available == False`) while its pure-Python helpers remain callable.

    python tests/test_anpr.py
    pytest  tests/test_anpr.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.anpr import AnprEngine, PlateReading              # noqa: E402


def _engine(region: str = "IN") -> AnprEngine:
    eng = AnprEngine({"weights": "does/not/exist.pt", "region": region})
    assert eng.available is False        # no model / no easyocr -> disabled
    return eng


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("already-valid Indian plate passes through unchanged")
def _():
    e = _engine()
    out, ok = e._coerce_in("WB06AB1234")
    assert (out, ok) == ("WB06AB1234", True)


@case("O/0 confusion in the RTO digits is corrected")
def _():
    e = _engine()
    out, ok = e._coerce_in("WBO6AB1234")     # letter O where a digit belongs
    assert out == "WB06AB1234" and ok is True


@case("digit-for-letter confusion in the state code is corrected")
def _():
    e = _engine()
    out, ok = e._coerce_in("W808AB1234")     # 8 -> B in the state code
    assert out == "WB08AB1234" and ok is True


@case("garbage that cannot be a plate is rejected")
def _():
    e = _engine()
    out, ok = e._coerce_in("XyZ")
    assert ok is False


@case("readings are isolated per camera by (cam_id, track_id)")
def _():
    e = _engine()
    now = time.time()
    with e._plates_lock:
        e._plates[(0, 1)] = PlateReading("AAA1111", 0.9, (0, 0, 1, 1), 0, 1, now, True)
        e._plates[(1, 1)] = PlateReading("BBB2222", 0.9, (0, 0, 1, 1), 1, 1, now, True)
    r0 = e.readings_for_cam(0)
    r1 = e.readings_for_cam(1)
    assert set(r0) == {1} and r0[1].text == "AAA1111"
    assert set(r1) == {1} and r1[1].text == "BBB2222"
    # flushing cam 0's track leaves cam 1's intact
    e.flush_track((0, 1))
    assert e.readings_for_cam(0) == {}
    assert e.readings_for_cam(1)[1].text == "BBB2222"


@case("submit_batch is a no-op (and never raises) while the engine is disabled")
def _():
    e = _engine()
    e.submit_batch([(0, None, [(0, 0, 10, 10, 1)])], tick=3)
    e.submit(0, None, [(0, 0, 10, 10, 1)], tick=3)
    assert e.status()["enabled"] is False


@case("_vote: positional majority picks the agreed characters")
def _():
    reads = [("WB06AB1234", 0.8, 0.0),
             ("WB06AB1234", 0.7, 0.1),
             ("WB06AB1284", 0.6, 0.2),   # one typo at position 8
             ("WB06AB1234", 0.9, 0.3)]
    text, agree = AnprEngine._vote(reads)
    assert text == "WB06AB1234", text
    assert 0.9 < agree <= 1.0, agree           # 9/10 positions unanimous, one 3/4


@case("_vote: a single-frame garbage read is outvoted by the track")
def _():
    reads = [("MH12DE1433", 0.9, 0.0), ("MH12DE1433", 0.8, 0.1),
             ("8H12DE1A33", 0.4, 0.2),         # blurry frame
             ("MH12DE1433", 0.85, 0.3)]
    text, _ = AnprEngine._vote(reads)
    assert text == "MH12DE1433", text


@case("_vote: modal length wins when reads disagree on length")
def _():
    reads = [("KA01AB1234", 0.8, 0), ("KA01AB1234", 0.8, 0),
             ("KA01AB123", 0.5, 0), ("KA01AB1234", 0.8, 0)]
    text, _ = AnprEngine._vote(reads)
    assert text == "KA01AB1234" and len(text) == 10


@case("OCR backend config is read (fast_plate default, vote knobs)")
def _():
    e = AnprEngine({"weights": "does/not/exist.pt"})
    assert e._ocr_backend == "fast_plate"
    assert e._vote_min == 3 and e._vote_max == 12
    e2 = AnprEngine({"weights": "x.pt", "ocr_backend": "easyocr",
                     "vote_min_reads": 5, "vote_max_reads": 9})
    assert e2._ocr_backend == "easyocr" and e2._vote_min == 5 and e2._vote_max == 9


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
