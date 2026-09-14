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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ibvap.geofence")

_TARGETS = {"person", "vehicle", "any"}
_DIRECTIONS = {"a2b", "b2a", "both"}
_KINDS = {"polygon", "line"}


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

    def wants(self, det) -> bool:
        if "any" in self.targets:
            return True
        return (("person" in self.targets and det.is_person) or
                ("vehicle" in self.targets and det.is_vehicle))

    def to_public(self) -> dict:
        """The subset the dashboard overlay and /status need."""
        return {"id": self.id, "cam_id": self.cam_id, "kind": self.kind,
                "points": [[round(x, 5), round(y, 5)] for x, y in self.points],
                "direction": self.direction, "label": self.label,
                "targets": sorted(self.targets), "enabled": self.enabled}


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
        self._by_cam: dict[int, list[Fence]] = {}
        # per-track state, keyed (cam_id, track_id) / (cam_id, fence_id, track_id)
        self._prev_ground: dict[tuple[int, int], tuple[float, float]] = {}
        self._inside: set[tuple[int, str, int]] = set()
        self._exit_streak: dict[tuple[int, str, int], int] = {}
        self._cooldown: dict[tuple[int, str, int], int] = {}

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

        fences = [f for f in self._by_cam.get(cam, []) if f.enabled]
        if not fences:
            return []

        frame = getattr(sr, "frame", None)
        h, w = (frame.shape[0], frame.shape[1]) if frame is not None else (1.0, 1.0)

        breaches: list[Breach] = []
        for d in sr.detections:
            tid = int(d.track_id)
            if tid < 0:
                continue
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
                            self._inside.add(key)
                            breaches.append(Breach(
                                cam, f, tid, d.class_name,
                                d.is_person, d.is_vehicle, g, None))
                    elif key in self._inside:
                        self._exit_streak[key] = self._exit_streak.get(key, 0) + 1
                        if self._exit_streak[key] >= self._exit_passes:
                            self._inside.discard(key)
                            self._exit_streak.pop(key, None)

                elif prev is not None:                      # line / polyline
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
                            breaches.append(Breach(
                                cam, f, tid, d.class_name,
                                d.is_person, d.is_vehicle, g, dirn))
                        break                               # one breach per fence per pass

            self._prev_ground[(cam, tid)] = g

        return breaches

    def prune(self, cam_id: int, live_track_ids) -> None:
        """Drop per-track state for track ids this camera no longer reports —
        mirrors PoseClassifier.flush_missing so the dicts can't grow unbounded
        and a reused track id can't inherit a stale 'inside' flag."""
        cam = int(cam_id)
        live = set(int(t) for t in live_track_ids)
        for k in [k for k in self._prev_ground if k[0] == cam and k[1] not in live]:
            del self._prev_ground[k]
        self._inside = {k for k in self._inside
                        if not (k[0] == cam and k[2] not in live)}
        for k in [k for k in self._exit_streak if k[0] == cam and k[2] not in live]:
            del self._exit_streak[k]
        for k in [k for k in self._cooldown if k[0] == cam and k[2] not in live]:
            del self._cooldown[k]
