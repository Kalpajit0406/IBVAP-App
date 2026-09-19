"""
geofence.py — operator-drawn virtual fences + intrusion detection.

The operator draws fences on a camera's own view in the dashboard FENCES panel:

  * polygon — a no-go area. Fires when a tracked target's GROUND POINT (the
              bottom-centre of its bounding box, i.e. where it stands) enters
              the area. Edge-triggered: one breach on entry, not one per frame
              while the target stays inside.
  * line    — a directional tripwire / fence line. Two or more points; each
              consecutive pair is a segment and crossing ANY of them fires,
              so a bent border fence can be traced with one fence. `direction`
              ("a2b", "b2a", "both") is relative to each segment's own A→B
              orientation. Debounced by `reentry_cooldown_passes` so a target
              loitering on the line doesn't re-fire.

Geometry is stored in normalised 0..1 image coordinates per `cam_id`, so a fence
survives a change of `ingest.normalise_*` or camera resolution. There is no
geo-projection — the platform has no camera calibration (see docs/GEOFENCE.md).

No GPU, no model, no OpenCV — pure geometry, fully unit-testable
(tests/test_geofence.py). `server.py`'s inference worker turns each Breach into
a CRITICAL alert + a hash-chained evidence record, the same override path the
weapon-ready posture and number-plate events use.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ibvap.geofence")

_TARGETS = {"person", "vehicle", "any"}
_DIRECTIONS = {"a2b", "b2a", "both"}
_KINDS = {"polygon", "line"}
_SEVERITIES = ("Info", "Low", "Medium", "High", "Critical")
_INBOUND = {"", "a2b", "b2a"}
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_MAX_LOITER_S = 86400.0          # a day; longer is a typo, not a policy
_MAX_FOLLOW_S = 600.0            # ten minutes between two crossings is not "following"


# ── geometry helpers (all coordinates normalised 0..1) ───────────────────────
def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def ground_point(bbox: tuple[float, float, float, float],
                 w: float, h: float) -> tuple[float, float]:
    """Where the target stands: bottom-centre of the pixel bbox, normalised."""
    x1, _y1, x2, y2 = bbox
    return (_clamp01(((x1 + x2) / 2.0) / max(w, 1.0)),
            _clamp01(y2 / max(h, 1.0)))


def point_in_polygon(pt: tuple[float, float],
                     poly: list[tuple[float, float]]) -> bool:
    """Even-odd ray casting. `poly` is an open ring (last point != first)."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _ccw(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, p3, p4) -> bool:
    """True iff segment p1-p2 properly crosses segment p3-p4."""
    d1, d2 = _ccw(p3, p4, p1), _ccw(p3, p4, p2)
    d3, d4 = _ccw(p1, p2, p3), _ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def side_of_line(a, b, p, eps: float = 1e-9) -> int:
    """Which side of the directed line a→b the point p is on: +1 / -1 / 0."""
    cross = _ccw(a, b, p)
    return 1 if cross > eps else -1 if cross < -eps else 0


def crossing_dir(prev, curr, a, b) -> Optional[str]:
    """Direction a segment prev→curr crossed the directed fence line a→b.

    "a2b"  — moved from the positive side of a→b to the negative side.
    "b2a"  — the reverse.
    None   — didn't cross the infinite line (same side, or touching it).
    The dashboard shows an arrowhead so the operator picks the side visually.
    """
    s0, s1 = side_of_line(a, b, prev), side_of_line(a, b, curr)
    if s0 == 0 or s1 == 0 or s0 == s1:
        return None
    return "a2b" if s0 > 0 else "b2a"


# ── data model ──────────────────────────────────────────────────────────────
@dataclass
class Fence:
    id: str
    cam_id: int
    kind: str                                   # "polygon" | "line"
    points: list[tuple[float, float]]           # normalised; polygon >=3, line ==2
    direction: str = "both"                     # line only
    targets: frozenset = field(default_factory=lambda: frozenset({"any"}))
    label: str = ""
    enabled: bool = True
    created_at: float = 0.0
    # Not every fence is an emergency. A perimeter wire at a BOP is Critical;
    # a counting line across an approach road is Info. One flat severity for
    # every zone is how an operator learns to ignore the alarm.
    severity: str = "Critical"
    # Raise a separate alert when someone stays inside this long (seconds).
    # 0 disables. "He has been at the fence eleven minutes" is a thing a post
    # actually acts on, and nothing could express it before.
    loiter_after_s: float = 0.0
    # Local-time arming window, "HH:MM". Equal values mean always armed. A
    # gate that is legitimately busy by day and forbidden after dark needs
    # this or it is turned off entirely.
    armed_from: str = ""
    armed_to: str = ""
    # Which crossing direction means "into our territory". There is no camera
    # calibration and therefore no true geo-projection, so this is what the
    # person who drew the line says it is — honest, and the only thing that
    # makes a count meaningful in a report.
    inbound: str = ""                           # "" | "a2b" | "b2a"
    # Lines only. Raise a `close_following` alert when a second target of the
    # same kind crosses this line the same way within this many seconds of
    # another — the vision-only form of tailgating. 0 = off. Vision cannot see
    # whether the first crosser was authorised, so this reports the *pattern*
    # (two crossings close together), not a verdict; at a gate where people
    # legitimately walk through in twos it will fire, which is why it is off
    # until someone turns it on for a specific fence.
    follow_window_s: float = 0.0

    def wants(self, det) -> bool:
        if "any" in self.targets:
            return True
        return (("person" in self.targets and det.is_person) or
                ("vehicle" in self.targets and det.is_vehicle))

    def armed_at(self, when: float) -> bool:
        """Is this fence live at `when` (epoch seconds, local time)?"""
        if not self.armed_from or not self.armed_to or self.armed_from == self.armed_to:
            return True
        try:
            f_h, f_m = (int(x) for x in self.armed_from.split(":"))
            t_h, t_m = (int(x) for x in self.armed_to.split(":"))
        except (ValueError, AttributeError):
            return True
        lt = time.localtime(when)
        now_m = lt.tm_hour * 60 + lt.tm_min
        start, end = f_h * 60 + f_m, t_h * 60 + t_m
        if start <= end:
            return start <= now_m < end
        return now_m >= start or now_m < end     # window crosses midnight

    def label_direction(self, dirn: Optional[str]) -> str:
        """'inbound' / 'outbound' when the operator has said which is which,
        else the raw geometric direction."""
        if not dirn or not self.inbound:
            return dirn or ""
        return "inbound" if dirn == self.inbound else "outbound"

    def to_public(self) -> dict:
        """The subset the dashboard overlay and /status need."""
        return {"id": self.id, "cam_id": self.cam_id, "kind": self.kind,
                "points": [[round(x, 5), round(y, 5)] for x, y in self.points],
                "direction": self.direction, "label": self.label,
                "targets": sorted(self.targets), "enabled": self.enabled,
                "severity": self.severity, "loiter_after_s": self.loiter_after_s,
                "armed_from": self.armed_from, "armed_to": self.armed_to,
                "inbound": self.inbound, "follow_window_s": self.follow_window_s}


@dataclass
class Breach:
    cam_id: int
    fence: Fence
    track_id: int
    class_name: str
    is_person: bool
    is_vehicle: bool
    ground_point: tuple[float, float]           # normalised
    direction: Optional[str]                    # set for line crossings
    event: str = "breach"                       # "breach" | "loiter" | "close_following"
    elapsed_s: float = 0.0                      # dwell time (loiter) / gap to the leader (close_following)
    leader_track: int = -1                      # close_following: the track that crossed just before


def _synth_id(cam_id: int, pts: list[tuple[float, float]]) -> str:
    h = abs(hash((int(cam_id), tuple(pts)))) % 0xFFFFFF
    return f"f_{h:06x}"


def _parse_fence(r: dict) -> Fence:
    pts = [(float(p[0]), float(p[1])) for p in r["points"]]
    tgt = frozenset(t for t in (r.get("targets") or ["any"]) if t in _TARGETS) \
        or frozenset({"any"})
    return Fence(
        id=str(r.get("id") or _synth_id(int(r["cam_id"]), pts)),
        cam_id=int(r["cam_id"]),
        kind=str(r["kind"]),
        points=pts,
        direction=str(r.get("direction", "both")),
        targets=tgt,
        label=str(r.get("label", "")),
        enabled=bool(r.get("enabled", True)),
        created_at=float(r.get("created_at", 0.0)),
        # `or` rather than a .get default: an explicit JSON null must fall back
        # to the default instead of becoming the string "None" or crashing
        # float(), which would silently discard the whole fence.
        severity=str(r.get("severity") or "Critical"),
        loiter_after_s=float(r.get("loiter_after_s") or 0.0),
        armed_from=str(r.get("armed_from") or ""),
        armed_to=str(r.get("armed_to") or ""),
        inbound=str(r.get("inbound") or ""),
        follow_window_s=float(r.get("follow_window_s") or 0.0),
    )


# ── engine ──────────────────────────────────────────────────────────────────
class GeoFenceEngine:
    """One instance, owned by the inference worker thread (single-threaded)."""

    def __init__(self, config: dict) -> None:
        gf = (config or {}).get("geofence", {}) or {}
        self._enabled = bool(gf.get("enabled", True))
        self._path = Path(gf.get("store", "data/fences.json"))
        self._cooldown_passes = max(1, int(gf.get("reentry_cooldown_passes", 8)))
        # Consecutive "outside" passes before a polygon breach is considered
        # ended — stops a target hugging the boundary from re-firing every pass.
        self._exit_passes = max(1, int(gf.get("exit_passes", 2)))
        # A single spurious detection whose ground point lands inside a polygon
        # used to raise Critical immediately. Requiring the track to have been
        # seen a few passes costs under a second and removes that whole class
        # of false alarm — which is what gets a system switched off.
        self._min_track_passes = max(1, int(gf.get("min_track_passes", 3)))
        self._min_box_frac = float(gf.get("min_box_height_frac", 0.0))
        self._by_cam: dict[int, list[Fence]] = {}
        # per-track state, keyed (cam_id, track_id) / (cam_id, fence_id, track_id)
        self._prev_ground: dict[tuple[int, int], tuple[float, float]] = {}
        # (cam, fence, track) -> epoch seconds the target entered. Was a bare
        # set; the timestamp is what makes dwell time answerable.
        self._inside: dict[tuple[int, str, int], float] = {}
        self._loitered: set[tuple[int, str, int]] = set()   # alerted once
        self._exit_streak: dict[tuple[int, str, int], int] = {}
        self._cooldown: dict[tuple[int, str, int], int] = {}
        self._seen: dict[tuple[int, int], int] = {}         # track maturity
        # (cam, fence_id, direction) -> count, since the last daily rollover
        self._crossings: dict[tuple[int, str, str], int] = {}
        # (cam, fence_id) -> recent crossings (t, track, is_person, direction),
        # for close-following detection. Bounded: only the last few matter.
        self._recent_cross: dict[tuple[int, str], deque] = {}
        self._counts_day = time.strftime("%Y-%m-%d")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def count(self) -> int:
        return sum(len(v) for v in self._by_cam.values())

    # ── loading ────────────────────────────────────────────────────────────
    def load(self) -> None:
        raw: list = []
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text("utf-8")).get("fences", [])
            except (ValueError, OSError) as e:
                logger.warning("could not read %s (%s) — starting with no fences",
                               self._path, e)
        self.set_fences(raw)

    def set_fences(self, raw: list) -> None:
        """Replace the whole fence set in memory (no disk write)."""
        errs = self.validate(raw)
        if errs:
            logger.warning("%d fence(s) rejected: %s", len(errs), "; ".join(errs[:4]))
        by_cam: dict[int, list[Fence]] = {}
        for r in raw:
            try:
                f = _parse_fence(r)
            except (KeyError, ValueError, TypeError):
                continue
            by_cam.setdefault(f.cam_id, []).append(f)
        self._by_cam = by_cam
        self._recent_cross = {}          # a redrawn fence starts with no history
        logger.info("fence set: %d fence(s) across %d camera(s)",
                    self.count, len(by_cam))

    @staticmethod
    def validate(raw) -> list[str]:
        """Return a list of human-readable errors; [] means the set is usable."""
        if not isinstance(raw, list):
            return ["fences must be a list"]
        errs: list[str] = []
        for i, r in enumerate(raw):
            p = f"fence[{i}]"
            if not isinstance(r, dict):
                errs.append(f"{p}: not an object")
                continue
            if r.get("kind") not in _KINDS:
                errs.append(f"{p}: kind must be one of {sorted(_KINDS)}")
            pts = r.get("points")
            ok_pts = isinstance(pts, list) and all(
                isinstance(q, (list, tuple)) and len(q) == 2 and
                all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in q)
                for q in pts)
            if not ok_pts:
                errs.append(f"{p}: points must be a list of [x,y] with 0<=x,y<=1")
            elif r.get("kind") == "polygon" and len(pts) < 3:
                errs.append(f"{p}: a polygon needs at least 3 points")
            elif r.get("kind") == "line" and len(pts) < 2:
                errs.append(f"{p}: a line needs at least 2 points")
            if r.get("direction", "both") not in _DIRECTIONS:
                errs.append(f"{p}: direction must be one of {sorted(_DIRECTIONS)}")
            tgt = r.get("targets", ["any"])
            if not isinstance(tgt, list) or not set(tgt) <= _TARGETS:
                errs.append(f"{p}: targets must be a subset of {sorted(_TARGETS)}")
            try:
                int(r["cam_id"])
            except (KeyError, TypeError, ValueError):
                errs.append(f"{p}: cam_id must be an integer")
            errs.extend(GeoFenceEngine._validate_settings(r, p))
        return errs

    @staticmethod
    def _validate_settings(r: dict, p: str) -> list[str]:
        """The per-fence behaviour fields. Absent and JSON null both mean
        "unset" — a client that does not know a field must not be rejected for
        omitting it, but a client that sends garbage must be told, because
        _parse_fence would otherwise drop the whole fence without a word."""
        errs: list[str] = []
        sev = r.get("severity")
        if sev is not None and sev not in _SEVERITIES:
            errs.append(f"{p}: severity must be one of {list(_SEVERITIES)}")
        loiter = r.get("loiter_after_s")
        if loiter is not None:
            # bool is an int in Python; `true` is not a number of seconds.
            if (isinstance(loiter, bool) or not isinstance(loiter, (int, float))
                    or not 0.0 <= float(loiter) <= _MAX_LOITER_S):
                errs.append(f"{p}: loiter_after_s must be a number of seconds "
                            f"between 0 and {int(_MAX_LOITER_S)}")
        frm, to = r.get("armed_from") or "", r.get("armed_to") or ""
        for name, val in (("armed_from", frm), ("armed_to", to)):
            if val and not (isinstance(val, str) and _HHMM.match(val)):
                errs.append(f"{p}: {name} must be a 24-hour HH:MM time")
        if bool(frm) != bool(to):
            # armed_at() treats a half-set window as "always armed", so a fence
            # meant to be night-only would silently stay live all day.
            errs.append(f"{p}: armed_from and armed_to must be set together")
        inbound = r.get("inbound")
        if inbound is not None and inbound not in _INBOUND:
            errs.append(f"{p}: inbound must be one of {sorted(_INBOUND)}")
        follow = r.get("follow_window_s")
        if follow is not None:
            if (isinstance(follow, bool) or not isinstance(follow, (int, float))
                    or not 0.0 <= float(follow) <= _MAX_FOLLOW_S):
                errs.append(f"{p}: follow_window_s must be a number of seconds "
                            f"between 0 and {int(_MAX_FOLLOW_S)}")
        return errs

    # ── query ──────────────────────────────────────────────────────────────
    def fences_for(self, cam_id: int) -> list[Fence]:
        return self._by_cam.get(int(cam_id), [])

    def public_for(self, cam_id: int) -> list[dict]:
        return [f.to_public() for f in self._by_cam.get(int(cam_id), []) if f.enabled]

    def active_zones(self, cam_id: int) -> dict[str, bool]:
        """fence_id -> is any track currently breaching it. Holds between passes
        (a polygon stays hot while someone is inside; a tripwire stays hot for
        the cooldown window after a crossing) so the overlay doesn't flicker."""
        cam = int(cam_id)
        out = {f.id: False for f in self._by_cam.get(cam, []) if f.enabled}
        for (c, fid, _tid) in self._inside:
            if c == cam and fid in out:
                out[fid] = True
        for (c, fid, _tid), n in self._cooldown.items():
            if c == cam and fid in out and n > 0:
                out[fid] = True
        return out

    # ── evaluation (once per detection pass per camera) ────────────────────
    def evaluate(self, sr) -> list[Breach]:
        """Return only the breaches that are NEW this pass (edge-triggered)."""
        cam = int(sr.cam_id)

        # age this camera's tripwire cooldowns by one pass
        for k in [k for k in self._cooldown if k[0] == cam]:
            self._cooldown[k] -= 1
            if self._cooldown[k] <= 0:
                del self._cooldown[k]

        now = time.time()
        self._roll_counts(now)
        fences = [f for f in self._by_cam.get(cam, [])
                  if f.enabled and f.armed_at(now)]
        if not fences:
            return []

        frame = getattr(sr, "frame", None)
        h, w = (frame.shape[0], frame.shape[1]) if frame is not None else (1.0, 1.0)

        breaches: list[Breach] = []
        for d in sr.detections:
            tid = int(d.track_id)
            if tid < 0:
                continue
            seen = self._seen[(cam, tid)] = self._seen.get((cam, tid), 0) + 1
            mature = seen >= self._min_track_passes
            if self._min_box_frac > 0 and frame is not None:
                y1, y2 = d.bbox[1], d.bbox[3]
                if (y2 - y1) < self._min_box_frac * h:
                    mature = False
            g = ground_point(d.bbox, w, h)
            prev = self._prev_ground.get((cam, tid))

            for f in fences:
                if not f.wants(d):
                    continue
                key = (cam, f.id, tid)

                if f.kind == "polygon":
                    if point_in_polygon(g, f.points):
                        self._exit_streak.pop(key, None)
                        if key not in self._inside:
                            if not mature:
                                continue      # too new to trust; no state either
                            self._inside[key] = now
                            breaches.append(Breach(
                                cam, f, tid, d.class_name,
                                d.is_person, d.is_vehicle, g, None))
                        elif (f.loiter_after_s > 0 and key not in self._loitered
                                and now - self._inside[key] >= f.loiter_after_s):
                            # Still inside, and has been for long enough to stop
                            # being an intrusion and start being a loiterer.
                            self._loitered.add(key)
                            breaches.append(Breach(
                                cam, f, tid, d.class_name,
                                d.is_person, d.is_vehicle, g, None,
                                event="loiter",
                                elapsed_s=round(now - self._inside[key], 1)))
                    elif key in self._inside:
                        self._exit_streak[key] = self._exit_streak.get(key, 0) + 1
                        if self._exit_streak[key] >= self._exit_passes:
                            self._inside.pop(key, None)
                            self._loitered.discard(key)
                            self._exit_streak.pop(key, None)

                elif prev is not None and mature:            # line / polyline
                    pts = f.points
                    for i in range(len(pts) - 1):
                        a, b = pts[i], pts[i + 1]
                        if not segments_intersect(prev, g, a, b):
                            continue
                        dirn = crossing_dir(prev, g, a, b)
                        if (dirn is not None
                                and f.direction in ("both", dirn)
                                and key not in self._cooldown):
                            self._cooldown[key] = self._cooldown_passes
                            ckey = (cam, f.id, f.label_direction(dirn) or dirn)
                            self._crossings[ckey] = self._crossings.get(ckey, 0) + 1
                            breaches.append(Breach(
                                cam, f, tid, d.class_name,
                                d.is_person, d.is_vehicle, g, dirn))
                            if f.follow_window_s > 0:
                                lead = self._leader(cam, f, tid, d.is_person,
                                                    dirn, now)
                                if lead is not None:
                                    breaches.append(Breach(
                                        cam, f, tid, d.class_name,
                                        d.is_person, d.is_vehicle, g, dirn,
                                        event="close_following",
                                        elapsed_s=round(now - lead[0], 2),
                                        leader_track=lead[1]))
                                self._recent_cross.setdefault(
                                    (cam, f.id), deque(maxlen=16)).append(
                                        (now, tid, d.is_person, dirn))
                        break                               # one breach per fence per pass

            self._prev_ground[(cam, tid)] = g

        return breaches

    def _leader(self, cam: int, f: Fence, tid: int, is_person: bool,
                dirn: str, now: float):
        """The most recent crossing that makes this one 'close following', or
        None. It must be a DIFFERENT track (one person crossing back and forth
        is not following themselves), the same kind of target (a person walking
        through behind a truck is not tailgating), and the same direction (two
        people passing each other in opposite directions are not following)."""
        for e in reversed(self._recent_cross.get((cam, f.id), ())):
            t0, ltid, lperson, ldir = e
            if now - t0 > f.follow_window_s:
                break                                       # older entries are older still
            if ltid != tid and lperson == is_person and ldir == dirn:
                return e
        return None

    # ── crossing tallies ────────────────────────────────────────────────────
    def _roll_counts(self, now: float) -> None:
        """Counts are per day. 'Six people crossed north-to-south tonight' is
        the report a post files; a running total since boot is not."""
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if day != self._counts_day:
            self._counts_day, self._crossings = day, {}

    def crossing_counts(self) -> dict:
        """{cam_id: {fence_id: {direction: n}}} for today, plus the date."""
        out: dict = {}
        for (cam, fid, dirn), n in sorted(self._crossings.items()):
            out.setdefault(str(cam), {}).setdefault(fid, {})[dirn] = n
        return {"date": self._counts_day, "by_camera": out}

    def dwelling(self, now: float | None = None) -> list[dict]:
        """Who is inside a zone right now, and for how long — the live answer
        to 'is anyone at the fence?' rather than a record that they once were."""
        now = time.time() if now is None else now
        return [{"cam_id": cam, "fence_id": fid, "track_id": tid,
                 "elapsed_s": round(now - since, 1),
                 "alerted": (cam, fid, tid) in self._loitered}
                for (cam, fid, tid), since in sorted(self._inside.items())]

    def prune(self, cam_id: int, live_track_ids) -> None:
        """Drop per-track state for track ids this camera no longer reports —
        mirrors PoseClassifier.flush_missing so the dicts can't grow unbounded
        and a reused track id can't inherit a stale 'inside' flag."""
        cam = int(cam_id)
        live = set(int(t) for t in live_track_ids)
        for k in [k for k in self._prev_ground if k[0] == cam and k[1] not in live]:
            del self._prev_ground[k]
        self._inside = {k: v for k, v in self._inside.items()
                        if not (k[0] == cam and k[2] not in live)}
        self._loitered = {k for k in self._loitered
                          if not (k[0] == cam and k[2] not in live)}
        for k in [k for k in self._exit_streak if k[0] == cam and k[2] not in live]:
            del self._exit_streak[k]
        for k in [k for k in self._cooldown if k[0] == cam and k[2] not in live]:
            del self._cooldown[k]
        for k in [k for k in self._seen if k[0] == cam and k[1] not in live]:
            del self._seen[k]
