"""
test_face.py -- FaceEngine construction/self-disable behaviour. No GPU: the
engine is pointed at missing weights/gallery paths so it stays disabled
(`available == False`) while still being safely constructible and callable.

    python tests/test_face.py
    pytest  tests/test_face.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.face import FaceEngine, FaceHit    # noqa: E402

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("missing gallery/weights -> disabled, but constructible and .identify_batch is a no-op")
def _():
    eng = FaceEngine({"gallery": "does/not/exist.json",
                      "model_root": "does/not/exist"}, device="cpu")
    assert eng.available is False
    assert eng.identify_batch([(0, None, [(0, 0, 10, 10, 1)])]) == {}
    assert eng.status()["enabled"] is False
    eng.stop()   # must not raise


@case("gallery present but weights missing -> still disabled")
def _():
    d = tempfile.mkdtemp()
    try:
        gpath = Path(d) / "gallery.json"
        gpath.write_text(json.dumps({
            "match_threshold": 0.38,
            "identities": [{"id": "a", "name": "A", "embedding": [0.0] * 512}],
        }))
        eng = FaceEngine({"gallery": str(gpath), "model_root": str(Path(d) / "nomodels")},
                         device="cpu")
        assert eng.available is False
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


@case("gallery with zero identities -> disabled even if the file parses")
def _():
    d = tempfile.mkdtemp()
    try:
        gpath = Path(d) / "gallery.json"
        gpath.write_text(json.dumps({"match_threshold": 0.38, "identities": []}))
        # weights dir also doesn't exist — either reason alone would disable it;
        # this exercises the empty-gallery branch specifically once the
        # existence check passes, by also creating an (empty, invalid) pack dir
        (Path(d) / "models" / "ibvap_face").mkdir(parents=True)
        (Path(d) / "models" / "ibvap_face" / "det_10g.onnx").write_bytes(b"")
        (Path(d) / "models" / "ibvap_face" / "w600k_r50.onnx").write_bytes(b"")
        eng = FaceEngine({"gallery": str(gpath), "model_root": str(d)}, device="cpu")
        assert eng.available is False
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


@case("malformed gallery JSON -> disabled, no raise")
def _():
    d = tempfile.mkdtemp()
    try:
        gpath = Path(d) / "gallery.json"
        gpath.write_text("{not valid json")
        # also satisfy the weights-exist check so this actually reaches (and
        # exercises) the JSON-parse branch, not an earlier "missing" one
        pack = Path(d) / "models" / "ibvap_face"
        pack.mkdir(parents=True)
        (pack / "det_10g.onnx").write_bytes(b"")
        (pack / "w600k_r50.onnx").write_bytes(b"")
        eng = FaceEngine({"gallery": str(gpath), "model_root": str(d)}, device="cpu")
        assert eng.available is False
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


@case("FaceHit dataclass round-trips its fields")
def _():
    h = FaceHit(person_track=3, bbox=(1, 2, 3, 4), det_score=0.9,
               matched_id="alice", matched_name="Alice", similarity=0.71,
               on_watchlist=True)
    assert h.person_track == 3 and h.bbox == (1, 2, 3, 4)
    assert h.matched_name == "Alice" and h.on_watchlist is True


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
