from __future__ import annotations

import datetime
from dataclasses import dataclass

from .detector import StreamResult


@dataclass
class RiskAssessment:
    cam_id: int
    score: float        # 0–100
    level: str          # "Normal" | "High" | "Critical"
    zone: float
    time_of_day: float
    behaviour: float
    weapon: bool = False       # a weapon threat (AIM posture OR a confirmed gun) this frame
    armed: bool = False        # AIM posture AND a confirmed gun on the SAME track
    weapon_tier: str = ""      # "armed" | "gun" | "posture" | ""  — drives the dashboard banner


class RiskEngine:
    """
    Weighted risk scoring:  score = (zone×0.4 + time×0.2 + behaviour×0.4) × 100
    Each component returns a value in [0, 1].
    """

    def __init__(self, config: dict) -> None:
        r = config["risk"]
        self._crit = r["threshold_critical"]
        # People / vehicles merely being present — however many, at night, in a
        # sensitive zone, with odd posture — is an elevated (High) situation, not
        # an alarm. Critical is reserved for weapon threats and fence breaches
        # (which override the level elsewhere). Set true to restore the old
        # score-only escalation.
        self._score_critical = bool(r.get("score_can_reach_critical", False))
        # The AIM rule is a 2-D skeleton guess with no view of any weapon (two
        # hands on a phone can trip it), so on its own it raises High. A gun the
        # object model confirmed — alone or with AIM — is still Critical.
        self._aim_only_critical = str(r.get("posture_aim_level", "High")).lower() == "critical"
        self._high = r["threshold_high"]
        w = r["weights"]
        self._w_zone = w["zone"]
        self._w_time = w["time_of_day"]
        self._w_beh = w["behaviour"]
        self._stream_cfg = {s["id"]: s for s in config["streams"]}

    def assess(self, sr: StreamResult) -> RiskAssessment:
        zone = self._zone(sr)
        tod = self._time_of_day()
        beh = self._behaviour(sr)

        raw = zone * self._w_zone + tod * self._w_time + beh * self._w_beh
        score = min(100.0, max(0.0, raw * 100))

        if score >= self._crit and self._score_critical:
            level = "Critical"
        elif score >= self._high:
            level = "High"
        else:
            level = "Normal"

        # A weapon threat is the highest there is — never gated by the weighted
        # formula (which only reaches Critical at night). Three tiers, fused
        # from the 2-D AIM posture heuristic (src/posture.py Rule 5) and the
        # trained gun detector (src/weapon.py):
        #   armed  = AIM posture AND a confirmed gun on the SAME person track
        #   gun    = a confirmed gun held by a tracked person
        #   posture= AIM posture only (no object confirmation)
        aim_tracks = {
            d.track_id for d in sr.detections
            if d.is_person and d.posture is not None and d.posture.chest_aim
        }
        gun_tracks = {
            h.person_track for h in (getattr(sr, "weapons", None) or [])
            if getattr(h, "confirmed", False) and getattr(h, "person_track", -1) >= 0
        }
        armed = bool(aim_tracks & gun_tracks)
        weapon = bool(aim_tracks) or bool(gun_tracks)

        if armed:
            level, score, tier = "Critical", max(score, 99.0), "armed"
        elif gun_tracks:
            level, score, tier = "Critical", max(score, 94.0), "gun"
        elif aim_tracks and self._aim_only_critical:
            level, score, tier = "Critical", max(score, 92.0), "posture"
        elif aim_tracks:
            level, score, tier = "High", max(score, min(self._high + 15.0, 69.0)), "posture"
            weapon = False       # unconfirmed: no weapon alarm, just an elevated camera
        else:
            tier = ""

        return RiskAssessment(sr.cam_id, score, level, zone, tod, beh,
                              weapon, armed, tier)

    # ── Component scorers ─────────────────────────────────────────────────────

    def _zone(self, sr: StreamResult) -> float:
        sensitivity = self._stream_cfg.get(sr.cam_id, {}).get("zone_sensitivity", 0.5)
        return sensitivity if (sr.person_count > 0 or sr.vehicle_count > 0) else 0.0

    def _time_of_day(self) -> float:
        hour = datetime.datetime.now().hour
        if 22 <= hour or hour < 5:
            return 1.0    # night
        if 5 <= hour < 7 or 20 <= hour < 22:
            return 0.6    # twilight
        return 0.2        # daytime

    def _behaviour(self, sr: StreamResult) -> float:
        """
        Combines person/vehicle count with per-person posture anomalies.
        PostureFlags are attached to Detection.posture by the detector.
        Falls back gracefully to count-only when pose is disabled.
        """
        count = sr.person_count + sr.vehicle_count
        if count == 0:
            return 0.0

        # Base score from head count
        if count == 1:
            base = 0.3
        elif count <= 3:
            base = 0.6
        else:
            base = 0.8

        # Posture boost: highest anomaly weight among all persons this frame
        posture_boost = max(
            (d.posture.risk_weight
             for d in sr.detections
             if d.is_person and d.posture is not None),
            default=0.0,
        )

        # Posture can only push the score up, never down
        return min(1.0, base + posture_boost * 0.5)
