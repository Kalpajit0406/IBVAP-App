"""
test_weapon.py -- WeaponDetector association + temporal vote, and the
train_weapon.py label remap. No model, no GPU: WeaponDetector is built with a
missing weights path (so `.available` is False) but every threshold and all the
per-track state are initialised, and the pure helpers are exercised directly.

    python tests/test_weapon.py
    pytest  tests/test_weapon.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.weapon import WeaponDetector, WeaponHit, _iou, _centre_in   # noqa: E402
from training import train_weapon                             # noqa: E402

WCFG = {
    "weights": "models/__does_not_exist__.pt",
    "min_conf": 0.30, "confirm_conf": 0.40, "min_box_area": 400, "person_iou_min": 0.05,
    "hold_hits": 2, "hold_window": 4, "detect_every": 2, "imgsz": 640,
}

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def _wd() -> WeaponDetector:
    wd = WeaponDetector(dict(WCFG), device="cpu")
    assert wd.available is False        # no weights → disabled, but usable helpers
    return wd


@case("_iou: overlap and disjoint")
def _():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0
    assert 0.1 < _iou((0, 0, 10, 10), (5, 5, 15, 15)) < 0.2


@case("_centre_in: gun just outside a person box still counts (grown silhouette)")
def _():
    person = (100, 100, 200, 400)
    assert _centre_in((150, 240, 170, 260), person)          # dead centre
    assert _centre_in((92, 240, 108, 260), person)           # just off the left edge
    assert not _centre_in((0, 0, 20, 20), person)            # nowhere near


@case("_associate: picks the overlapping person track, -1 when free-floating")
def _():
    wd = _wd()
    persons = [(100, 100, 200, 400, 7), (600, 100, 700, 400, 9)]
    assert wd._associate((150, 250, 180, 280), persons)[0] == 7
    assert wd._associate((650, 250, 680, 280), persons)[0] == 9
    assert wd._associate((400, 50, 430, 90), persons)[0] == -1
    # a background object that only touches the top of the box is not held
    assert wd._associate((60, 40, 260, 120), persons)[0] == -1


@case("_associate: with visible wrists, the gun must be at a hand")
def _():
    wd = _wd()
    at_hand = [(100, 100, 200, 400, 7, False, [(165, 265)])]
    hands_elsewhere = [(100, 100, 200, 400, 7, False, [(110, 380), (190, 380)])]
    unknown = [(100, 100, 200, 400, 7, False, [])]
    gun = (150, 250, 180, 280)
    assert wd._associate(gun, at_hand)[0] == 7
    assert wd._associate(gun, hands_elsewhere)[0] == -1
    assert wd._associate(gun, unknown)[0] == 7          # no pose -> box rule only


@case("_classify_box: a silhouette-sized or frame-sized box is rejected")
def _():
    wd = _wd()
    persons = [(100, 100, 200, 400, 1)]
    assert wd._classify_box(0, (100, 120, 200, 390), 0.9, "gun", persons, 1.0) is None
    assert wd._classify_box(0, (0, 0, 700, 400), 0.9, "gun", [], 1.0,
                            frame_area=1280 * 720) is None


@case("_classify_box: min_conf and min_box_area reject before anything else")
def _():
    wd = _wd()
    persons = [(100, 100, 200, 400, 1)]
    assert wd._classify_box(0, (150, 250, 180, 300), 0.20, "gun", persons, 1.0) is None
    assert wd._classify_box(0, (150, 250, 155, 255), 0.90, "gun", persons, 1.0) is None
    assert wd._n_raw == 0


PERSON = [(100, 100, 200, 400, 4)]
GUN = (150, 250, 185, 285)                        # ~35x35 = 1225 px^2 > min_area


def _pass(wd, conf=None, persons=PERSON):
    """One weapon pass; conf=None means the model saw no gun this pass."""
    wd.begin_pass()
    hit = None if conf is None else wd._classify_box(0, GUN, conf, "gun", persons, 1.0)
    wd._expire_history()
    return hit


@case("temporal vote: confirmed on the hold_hits-th held pass")
def _():
    wd = _wd()
    seen = [_pass(wd, 0.8).confirmed for _ in range(3)]
    assert seen == [False, True, True], seen


@case("a missed pass between hits does NOT reset the vote (2 of the last 4)")
def _():
    wd = _wd()
    assert _pass(wd, 0.8).confirmed is False
    _pass(wd, None)                               # model blinked
    assert _pass(wd, 0.8).confirmed is True


@case("hits further apart than the window never confirm")
def _():
    wd = _wd()
    for _ in range(3):
        assert _pass(wd, 0.8).confirmed is False
        for _ in range(4):
            _pass(wd, None)
    assert wd.drain_events() == []


@case("low-confidence hits alone never confirm (confirm_conf)")
def _():
    wd = _wd()
    seen = [_pass(wd, 0.33).confirmed for _ in range(5)]
    assert not any(seen), seen


@case("a held gun on a person in an AIM posture confirms on the first pass")
def _():
    wd = _wd()
    aiming = [(100, 100, 200, 400, 4, True)]
    assert _pass(wd, 0.33, persons=aiming).confirmed is True


@case("an older config's hold_frames still sets the vote size")
def _():
    wd = WeaponDetector({"weights": "models/__nope__.pt", "hold_frames": 3}, device="cpu")
    seen = [_pass(wd, 0.8).confirmed for _ in range(3)]
    assert seen == [False, False, True], seen


@case("a free-floating gun never confirms and emits no event")
def _():
    wd = _wd()
    for t in range(6):
        wd.begin_pass()
        h = wd._classify_box(0, (400, 50, 440, 95), 0.9, "gun", [], float(t))
        assert h.confirmed is False and h.person_track == -1
    assert wd.drain_events() == []


@case("exactly one WeaponEvent is emitted when a track first confirms")
def _():
    wd = _wd()
    for _ in range(5):
        _pass(wd, 0.8)
    evs = wd.drain_events()
    assert len(evs) == 1 and evs[0].track_id == 4 and evs[0].tier == "gun", evs
    assert wd.drain_events() == []                # drained


@case("flush_track clears the vote so it restarts")
def _():
    wd = _wd()
    _pass(wd, 0.8)
    wd.flush_track((0, 4))
    assert _pass(wd, 0.8).confirmed is False


@case("train_weapon._gun_class: two-class labels on disk beat a stale gun=0 marker")
def _():
    with tempfile.TemporaryDirectory() as d:
        lab = Path(d) / "labels" / "val"
        lab.mkdir(parents=True)
        (Path(d) / "labels" / ".gun_class").write_text("0")
        (lab / "a.txt").write_text("0 0.5 0.5 0.3 0.8\n1 0.4 0.4 0.1 0.1\n")
        assert train_weapon._gun_class(Path(d)) == 1
        (lab / "a.txt").write_text("0 0.4 0.4 0.1 0.1\n")      # genuinely gun-only
        assert train_weapon._gun_class(Path(d)) == 0


@case("train_weapon._convert_label: keep gun (cls 1 -> 0), drop person (cls 0)")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.txt"
        p.write_text("0 0.5 0.5 0.2 0.8\n1 0.60 0.40 0.10 0.12\n0 0.1 0.1 0.1 0.1\n")
        text, has_gun = train_weapon._convert_label(p)
        assert has_gun is True
        lines = text.strip().splitlines()
        assert lines == ["0 0.60 0.40 0.10 0.12"], lines

        empty = Path(d) / "y.txt"
        empty.write_text("0 0.5 0.5 0.2 0.8\n")            # person only
        text2, has_gun2 = train_weapon._convert_label(empty)
        assert has_gun2 is False and text2 == ""


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
