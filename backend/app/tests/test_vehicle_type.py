"""
test_vehicle_type.py -- VehicleTypeClassifier's coarse-COCO fallback path (no
GPU, no network: weights are pointed at a path that doesn't exist, so the
engine never touches ultralytics/torch).

    python tests/test_vehicle_type.py        # prints PASS/FAIL, exits 1 on failure
    pytest  tests/test_vehicle_type.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                    # noqa: E402

from ibvap.vehicle_type import CLASSES, VehicleTypeClassifier          # noqa: E402

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def _crop():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _cfg(**over) -> dict:
    base = {"weights": "models/does_not_exist_vehicle_type.pt"}
    base.update(over)
    return base


@case("CLASSES has exactly the 8 documented classes, in order")
def _():
    assert CLASSES == ["car", "pickup_truck", "truck", "jeep", "two_wheeler",
                       "tanker", "van", "auto_rickshaw"]


@case("missing weights -> available is False")
def _():
    c = VehicleTypeClassifier(_cfg())
    assert c.available is False


@case("fallback maps COCO car/motorcycle/bus/truck, unknown class -> 'unknown'")
def _():
    c = VehicleTypeClassifier(_cfg())
    items = [(0, 1, _crop(), 2), (0, 2, _crop(), 3),
             (0, 3, _crop(), 5), (0, 4, _crop(), 7),
             (0, 5, _crop(), 999)]
    out = c.classify_batch(items)
    assert out[(0, 1)].label == "car"
    assert out[(0, 2)].label == "two_wheeler"
    assert out[(0, 3)].label == "van"
    assert out[(0, 4)].label == "truck"
    assert out[(0, 5)].label == "unknown"
    for r in out.values():
        assert r.conf == 0.0 and r.fallback is True


@case("fallback never produces jeep/pickup_truck/tanker/auto_rickshaw")
def _():
    c = VehicleTypeClassifier(_cfg())
    out = c.classify_batch([(0, i, _crop(), cid)
                            for i, cid in enumerate([2, 3, 5, 7])])
    unreachable = {"jeep", "pickup_truck", "tanker", "auto_rickshaw"}
    assert not any(r.label in unreachable for r in out.values())


@case("fallback_coco: false -> 'unknown' for everything, still conf 0 / fallback True")
def _():
    c = VehicleTypeClassifier(_cfg(fallback_coco=False))
    out = c.classify_batch([(0, 1, _crop(), 2)])
    assert out[(0, 1)].label == "unknown"
    assert out[(0, 1)].conf == 0.0 and out[(0, 1)].fallback is True


@case("empty batch -> empty result, no crash")
def _():
    c = VehicleTypeClassifier(_cfg())
    assert c.classify_batch([]) == {}


@case("status() reports disabled engine cleanly")
def _():
    c = VehicleTypeClassifier(_cfg())
    s = c.status()
    assert s["enabled"] is False
    assert s["classes"] == CLASSES


@case("labels come from the checkpoint's own names, not CLASSES' order")
def _():
    """Regression: ultralytics sorts ImageFolder class dirs alphabetically, so a
    trained checkpoint's index order is NOT the order CLASSES is written in.
    Indexing CLASSES with the model's top1 mislabelled every prediction (index 1
    is 'car' to the model but 'pickup_truck' in CLASSES)."""
    alpha = sorted(CLASSES)                      # what ultralytics actually produces
    assert alpha != CLASSES, "test is meaningless if the orders already agree"

    c = VehicleTypeClassifier(_cfg())
    c.available = True                           # stub in a fake loaded model
    c._names = {i: n for i, n in enumerate(alpha)}
    c._min_conf = 0.1

    want_idx = alpha.index("car")
    c._model = types.SimpleNamespace(
        predict=lambda *a, **k: [types.SimpleNamespace(
            probs=types.SimpleNamespace(top1=want_idx, top1conf=0.95))])

    res = c.classify_batch([(0, 1, _crop(), 2)])[(0, 1)]
    assert res.label == "car", f"got {res.label!r} — index/name mapping is wrong"
    assert res.fallback is False
    assert c.status()["classes"] == alpha        # reports the checkpoint's list
    assert c.status()["taxonomy"] == CLASSES     # ...and the full taxonomy too


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
