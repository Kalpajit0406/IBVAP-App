"""
test_risk_engine.py -- RiskEngine scoring. No model, no GPU: hand-built
StreamResult / Detection objects.

    python tests/test_risk_engine.py
    pytest  tests/test_risk_engine.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.detector import Detection, StreamResult          # noqa: E402
from ibvap.posture import PostureFlags                       # noqa: E402
from ibvap.risk_engine import RiskEngine                     # noqa: E402
from ibvap.weapon import WeaponHit                           # noqa: E402

CFG = {
    "risk": {
        "threshold_critical": 70,
        "threshold_high": 50,
        "weights": {"zone": 0.40, "time_of_day": 0.20, "behaviour": 0.40},
    },
    "streams": [
        {"id": 0, "zone_sensitivity": 0.9},
        {"id": 1, "zone_sensitivity": 0.2},
    ],
}


def _person(tid: int, *, aim: bool = False, lying: bool = False) -> Detection:
    d = Detection(track_id=tid, class_id=0, class_name="person",
                  bbox=(0, 0, 10, 20), confidence=0.9,
                  is_person=True, is_vehicle=False)
    if aim or lying:
        d.posture = PostureFlags(track_id=tid, chest_aim=aim, lying=lying)
    return d


def _sr(cam_id: int, dets: list[Detection], weapons: list | None = None) -> StreamResult:
    return StreamResult(
        cam_id=cam_id, timestamp=0.0, frame=None, detections=dets,
        person_count=sum(d.is_person for d in dets),
        vehicle_count=sum(d.is_vehicle for d in dets),
        weapons=weapons or [],
    )


def _gun(track: int, confirmed: bool = True) -> WeaponHit:
    return WeaponHit(bbox=(0, 0, 5, 5), conf=0.8, cls_name="gun",
                     person_track=track, confirmed=confirmed)


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("empty scene -> Normal, low score")
def _(rk):
    ra = rk.assess(_sr(0, []))
    assert ra.level == "Normal" and ra.score < 50, ra


@case("AIM posture alone (no gun) -> High, not a weapon alarm")
def _(rk):
    ra = rk.assess(_sr(1, [_person(1, aim=True)]))     # cold zone, any time of day
    assert ra.level == "High", ra
    assert 50 <= ra.score < 70, ra
    assert ra.weapon is False and ra.weapon_tier == "posture", ra


@case("posture_aim_level: Critical restores the old AIM escalation")
def _(rk):
    cfg = {**CFG, "risk": {**CFG["risk"], "posture_aim_level": "Critical"}}
    ra = RiskEngine(cfg).assess(_sr(0, [_person(1, aim=True)]))
    assert ra.level == "Critical" and ra.score >= 92 and ra.weapon is True, ra


@case("people detected at night in a hot zone -> High at most, never Critical")
def _(rk):
    rk._time_of_day = lambda: 1.0
    ra = rk.assess(_sr(0, [_person(i, lying=(i == 0)) for i in range(5)]))
    assert ra.score >= 70, ra              # the weighted score is still reported
    assert ra.level == "High", ra
    assert ra.weapon is False


@case("score_can_reach_critical: true restores score-only Critical")
def _(rk):
    cfg = {**CFG, "risk": {**CFG["risk"], "score_can_reach_critical": True}}
    eng = RiskEngine(cfg)
    eng._time_of_day = lambda: 1.0
    ra = eng.assess(_sr(0, [_person(i) for i in range(5)]))
    assert ra.level == "Critical", ra


@case("lying person on a hot zone -> High, weapon flag stays off")
def _(rk):
    ra = rk.assess(_sr(0, [_person(1, lying=True)]))
    assert ra.level in ("Normal", "High"), ra
    assert ra.weapon is False


@case("zone sensitivity changes the score for the same scene")
def _(rk):
    hot = rk.assess(_sr(0, [_person(1)])).score      # zone_sensitivity 0.9
    cold = rk.assess(_sr(1, [_person(1)])).score     # zone_sensitivity 0.2
    assert hot > cold, (hot, cold)


@case("unknown camera id falls back to default sensitivity, no crash")
def _(rk):
    ra = rk.assess(_sr(99, [_person(1)]))
    assert ra.level in ("Normal", "High", "Critical")


@case("more people -> higher behaviour score")
def _(rk):
    one = rk.assess(_sr(0, [_person(1)])).behaviour
    four = rk.assess(_sr(0, [_person(i) for i in range(4)])).behaviour
    assert four > one, (one, four)


@case("a confirmed gun alone forces Critical, tier 'gun', score >= 94")
def _(rk):
    ra = rk.assess(_sr(1, [_person(1)], weapons=[_gun(1)]))     # cold zone
    assert ra.level == "Critical" and ra.score >= 94, ra
    assert ra.weapon is True and ra.armed is False
    assert ra.weapon_tier == "gun", ra


@case("an UNconfirmed gun never forces Critical and sets no weapon flag")
def _(rk):
    ra = rk.assess(_sr(1, [_person(1)], weapons=[_gun(1, confirmed=False)]))
    assert ra.weapon is False and ra.armed is False, ra
    assert ra.weapon_tier == "", ra


@case("AIM posture + confirmed gun on the SAME track -> armed, score 99")
def _(rk):
    ra = rk.assess(_sr(1, [_person(1, aim=True)], weapons=[_gun(1)]))
    assert ra.armed is True and ra.weapon is True, ra
    assert ra.weapon_tier == "armed" and ra.score >= 99, ra


@case("AIM on one track, gun on another -> weapon yes, armed no")
def _(rk):
    ra = rk.assess(_sr(1, [_person(1, aim=True), _person(2)], weapons=[_gun(2)]))
    assert ra.weapon is True and ra.armed is False, ra
    assert ra.weapon_tier in ("gun", "posture"), ra


def run() -> int:
    failed = 0
    for name, fn in CASES:
        rk = RiskEngine(CFG)
        try:
            fn(rk)
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
