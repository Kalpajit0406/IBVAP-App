"""
behaviour.py — behaviour rules that read tracks, not pixels.

The posture heuristics look at one person's skeleton and the fence looks at one
line on the ground. Two things an SSB post actually asks about needed neither:
is someone RUNNING, and has a GROUP formed. Both are properties of how tracked
people move relative to the scene and to each other, so they are computed from
the tracks the detector already produces — no new model, no GPU cost.

(A third, "close-following crossing", is a property of a tripwire crossing and
so lives in geofence.py next to the crossing logic it depends on.)

What these are, honestly: heuristics. Nothing here has a measured false-alarm
rate yet, so every event defaults to Medium severity and none of them can reach
Critical on their own — the standing rule that a detected person alone is never
a Critical alarm applies to how they move just as much as to whether they are
there. Thresholds are in config.yaml `behaviour:` and are meant to be tuned on
real footage from the post.

Both rules work in BODY-HEIGHTS rather than pixels. A person 60 px tall near the
horizon and one 300 px tall at the fence cover very different pixel distances at
the same real speed; dividing by their own box height cancels most of that
perspective without needing camera calibration (which does not exist). A walking
adult is ~0.8 body-heights per second, a jog ~1.5, a run 2+.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger("ibvap.behaviour")

_SEVERITIES = ("Info", "Low", "Medium", "High", "Critical")


def _sev(value, default: str) -> str:
    return value if value in _SEVERITIES else default


@dataclass
class BehaviourEvent:
    kind: str                     # "running" | "group"
    cam_id: int
    key: int                      # track id (running) / group id (group): the
                                  # snapshot-cooldown key
    severity: str
    detail: str                   # short label for the snapshot filename / log
    details: dict = field(default_factory=dict)   # goes into the evidence record


@dataclass
class _Track:
    samples: deque = field(default_factory=deque)   # (t, x, y, h) inside the window
    seen: int = 0                 # detection passes this track has been on
    last_t: float = 0.0
    hot: int = 0                  # consecutive passes at/above the run speed
    armed: bool = True            # False after firing, until the person slows down
    alerted_at: float = -1e9
    running_until: float = -1e9   # overlay hold, so the label doesn't flicker
    speed: float = 0.0            # last measured, body-heights per second
    peak: float = 0.0             # highest measured, for tuning the threshold
    wipes: int = 0                # times the glitch guard restarted this history


@dataclass
class _Group:
    id: int
    members: set
    since: float
    last_seen: float
    peak: int
    alerted: bool = False


class BehaviourEngine:
    """One instance, owned by the inference worker thread (single-threaded).

    `evaluate(sr)` once per real detection pass per camera — never on the
    carried-forward frames in between, whose boxes are extrapolated and would
    make every track look perfectly smooth.
    """

    def __init__(self, config: dict) -> None:
        b = (config or {}).get("behaviour", {}) or {}
        self.enabled = bool(b.get("enabled", True))
        # A track must have been seen this many passes before it can trigger
        # anything. One spurious detection must not be able to form a "group".
        self._min_passes = max(1, int(b.get("min_track_passes", 3)))
        # Ignore people smaller than this fraction of frame height: at that
        # size a few pixels of box jitter is a large fraction of a body-height
        # and reads as motion.
        self._min_h_frac = float(b.get("min_box_height_frac", 0.06))
        self._track_grace_s = float(b.get("track_grace_s", 3.0))

        r = b.get("running") or {}
        self._run_on = bool(r.get("enabled", True))
        self._run_speed = float(r.get("speed_bh_s", 2.0))
        self._run_window = max(0.3, float(r.get("window_s", 1.0)))
        self._run_sustain = max(1, int(r.get("sustain_passes", 3)))
        self._run_glitch = float(r.get("glitch_bh_s", 6.0))
        self._run_cool = float(r.get("cooldown_s", 30.0))
        self._run_hold = float(r.get("overlay_hold_s", 1.5))
        self._run_sev = _sev(r.get("severity"), "Medium")

        g = b.get("group") or {}
        self._grp_on = bool(g.get("enabled", True))
        self._grp_radius = float(g.get("radius_bh", 1.5))
        self._grp_min = max(2, int(g.get("min_size", 3)))
        self._grp_min_s = float(g.get("min_duration_s", 3.0))
        self._grp_grace = float(g.get("grace_s", 2.0))
        self._grp_cool = float(g.get("cooldown_s", 60.0))
        self._grp_sev = _sev(g.get("severity"), "Medium")

        self._tracks: dict[tuple[int, int], _Track] = {}
        self._groups: dict[int, list[_Group]] = {}
        self._last_group_alert: dict[int, float] = {}
        self._group_seq = 0
        self._now: dict[int, float] = {}
        self._counts = {"running": 0, "group": 0}

    # ── one detection pass ──────────────────────────────────────────────────
    def evaluate(self, sr) -> list[BehaviourEvent]:
        if not self.enabled:
            return []
        cam, t = int(sr.cam_id), float(sr.timestamp)
        frame = getattr(sr, "frame", None)
        fh = float(frame.shape[0]) if frame is not None else 0.0
        self._now[cam] = t

        people = []                       # (tid, x, y, h, track)
        for d in sr.detections:
            if not d.is_person or d.track_id < 0:
                continue
            x1, y1, x2, y2 = d.bbox
            h = float(y2 - y1)
            if h <= 1.0 or (fh and h < self._min_h_frac * fh):
                continue
            tid = int(d.track_id)
            tr = self._tracks.get((cam, tid))
            if tr is None:
                tr = self._tracks[(cam, tid)] = _Track()
            tr.seen += 1
            tr.last_t = t
            # Ground point: bottom-centre, where the person stands.
            people.append((tid, (x1 + x2) / 2.0, float(y2), h, tr))

        out: list[BehaviourEvent] = []
        if self._run_on:
            out += self._running(cam, t, people)
        if self._grp_on:
            out += self._grouping(cam, t, people)
        return out

    # ── running ─────────────────────────────────────────────────────────────
    def _running(self, cam: int, t: float, people: list) -> list[BehaviourEvent]:
        out: list[BehaviourEvent] = []
        for tid, x, y, h, tr in people:
            s = tr.samples
            if s:
                pt, px, py, ph = s[-1]
                dt = t - pt
                if dt <= 1e-6:
                    continue              # the same frame seen twice
                step = math.hypot(x - px, y - py) / ((ph + h) / 2.0) / dt
                if step > self._run_glitch:
                    # Nobody covers this many body-heights per second. It is the
                    # tracker swapping two people or re-associating a lost box,
                    # and left in the window it would read as a sprint for a
                    # whole second. Start that track's history over.
                    s.clear()
                    tr.hot = 0
                    tr.wipes += 1
            s.append((t, x, y, h))
            while s and t - s[0][0] > self._run_window:
                s.popleft()

            # Speed is the NET displacement across the window, not the sum of
            # frame-to-frame steps: box jitter adds a little to every step and
            # would make a person standing still look like they are moving,
            # whereas over a full second it nets to almost nothing.
            span = t - s[0][0]
            speed = None
            if len(s) >= 3 and span >= 0.7 * self._run_window:
                mean_h = sum(p[3] for p in s) / len(s)
                speed = math.hypot(x - s[0][1], y - s[0][2]) / mean_h / span
            tr.speed = speed or 0.0
            tr.peak = max(tr.peak, tr.speed)

            if speed is not None and speed >= self._run_speed:
                tr.hot += 1
                tr.running_until = t + self._run_hold
            elif speed is None or speed < 0.8 * self._run_speed:
                tr.hot = 0
                tr.armed = True           # slowed down: may alert again later

            if (tr.hot >= self._run_sustain and tr.armed
                    and t - tr.alerted_at >= self._run_cool
                    and tr.seen >= self._min_passes):
                tr.alerted_at, tr.armed = t, False
                self._counts["running"] += 1
                out.append(BehaviourEvent(
                    "running", cam, tid, self._run_sev,
                    f"{speed:.1f}bh_s",
                    {"track_id": tid, "speed_bh_s": round(speed, 2),
                     "window_s": round(span, 2)}))
        return out

    # ── group formation ─────────────────────────────────────────────────────
    def _components(self, people: list) -> list[set]:
        """Groups of people linked by 'within radius_bh of each other',
        transitively — A near B and B near C is one group even if A and C are
        far apart, which is how a strung-out huddle or a queue actually looks."""
        n = len(people)
        if n < self._grp_min:
            return []
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            _, xi, yi, hi, _ = people[i]
            for j in range(i + 1, n):
                _, xj, yj, hj, _ = people[j]
                if math.hypot(xi - xj, yi - yj) / ((hi + hj) / 2.0) <= self._grp_radius:
                    parent[find(i)] = find(j)
        comps: dict[int, set] = {}
        for i in range(n):
            comps.setdefault(find(i), set()).add(people[i][0])
        return [c for c in comps.values() if len(c) >= self._grp_min]

    def _grouping(self, cam: int, t: float, people: list) -> list[BehaviourEvent]:
        mature = [p for p in people if p[4].seen >= self._min_passes]
        candidates = self._components(mature)
        groups = self._groups.setdefault(cam, [])

        # Carry each existing group's identity across passes by membership
        # overlap, so one person joining or leaving does not restart the clock
        # or raise a second alert for what is plainly the same gathering.
        unmatched = list(candidates)
        for g in groups:
            best, best_ov = None, 0.0
            for c in unmatched:
                ov = len(g.members & c) / max(1, min(len(g.members), len(c)))
                if ov > best_ov:
                    best, best_ov = c, ov
            if best is not None and best_ov >= 0.5:
                g.members, g.last_seen = set(best), t
                g.peak = max(g.peak, len(best))
                unmatched.remove(best)
        for c in unmatched:
            self._group_seq += 1
            groups.append(_Group(self._group_seq, set(c), t, t, len(c)))
        groups[:] = [g for g in groups if t - g.last_seen <= self._grp_grace]

        out: list[BehaviourEvent] = []
        for g in groups:
            if g.alerted or g.last_seen != t or t - g.since < self._grp_min_s:
                continue
            if t - self._last_group_alert.get(cam, -1e9) < self._grp_cool:
                continue                  # a gathering that flickers apart and
                                          # re-forms is one event, not a stream
            g.alerted = True
            self._last_group_alert[cam] = t
            self._counts["group"] += 1
            out.append(BehaviourEvent(
                "group", cam, g.id, self._grp_sev, f"{len(g.members)}p",
                {"group_id": g.id, "size": len(g.members),
                 "members": sorted(g.members),
                 "duration_s": round(t - g.since, 1)}))
        return out

    # ── housekeeping / observability ────────────────────────────────────────
    def prune(self, cam_id: int, live_track_ids=None) -> None:
        """Drop per-track state for people gone longer than the grace period.
        A grace period rather than dropping the instant a track misses one pass:
        a detection miss is routine, and losing the history would restart every
        speed window and group clock on it."""
        cam = int(cam_id)
        now = self._now.get(cam)
        live = {int(x) for x in (live_track_ids or ())}
        for k in [k for k, tr in self._tracks.items()
                  if k[0] == cam and k[1] not in live
                  and (now is None or now - tr.last_t > self._track_grace_s)]:
            del self._tracks[k]

    def overlay(self, cam_id: int) -> dict:
        """What the video overlay should mark right now: track ids currently
        running, and the members of each confirmed group."""
        cam = int(cam_id)
        now = self._now.get(cam)
        if now is None:
            return {"running": set(), "groups": []}
        return {
            "running": {tid for (c, tid), tr in self._tracks.items()
                        if c == cam and tr.running_until >= now},
            "groups": [{"id": g.id, "members": set(g.members)}
                       for g in self._groups.get(cam, [])
                       if g.alerted and now - g.last_seen <= self._grp_grace],
        }

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "running": {"enabled": self._run_on, "events": self._counts["running"],
                        "speed_bh_s": self._run_speed},
            "group": {"enabled": self._grp_on, "events": self._counts["group"],
                      "min_size": self._grp_min, "radius_bh": self._grp_radius,
                      "min_duration_s": self._grp_min_s},
            "tracked_people": len(self._tracks),
            # The people moving fastest right now. This is how an operator tunes
            # `running.speed_bh_s` on real footage: watch what walking, jogging
            # and running actually measure at THIS camera, then set the
            # threshold between them.
            "fastest": [
                {"cam": c, "track": tid, "speed_bh_s": round(tr.speed, 2),
                 "peak_bh_s": round(tr.peak, 2), "seen": tr.seen,
                 "history_restarts": tr.wipes}
                for (c, tid), tr in sorted(self._tracks.items(),
                                           key=lambda kv: -kv[1].peak)[:8]
            ],
        }
