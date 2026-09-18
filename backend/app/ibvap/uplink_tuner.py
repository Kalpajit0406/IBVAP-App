"""
uplink_tuner.py — pick each phone's send resolution / frame rate / JPEG quality
from the latency the server actually measures.

Why the server decides. The phone cannot see its own uplink: `ws.bufferedAmount`
goes to zero the moment the kernel accepts the bytes, while they sit for seconds
in the socket and Wi-Fi driver queues. The server, by contrast, knows the true
clock-synced capture→receive delay for every frame (`ws_capture.latency_ms`),
and — unlike any single phone — can see all of them at once, which matters
because they share one bottleneck.

Why rungs rather than continuous control. Changing the canvas size mid-stream is
visible and cheap to get wrong; a short ladder of vetted operating points is
easier to reason about and to explain to an operator watching the overlay.

Which knob gives way first. Frame rate, always. Face recognition needs *pixels
on the face* — a burst only needs a handful of good frames over several seconds,
so halving fps costs recognition almost nothing while buying bitrate linearly.
Halving resolution costs recognition range directly. So the ladder drops fps
before it drops resolution, and climbs back to resolution first.

The credit loop in static/camera.html already bounds latency by construction;
this only chooses how much detail fits inside that bound.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("ibvap.uplink")


@dataclass(frozen=True)
class Rung:
    w: int
    h: int
    fps: int
    q: float
    mbps: float          # rough offered bitrate, for the shared budget

    @property
    def label(self) -> str:
        return f"{self.w}x{self.h}@{self.fps}"


# Bitrates are measured-ish estimates for a typical indoor scene: bytes/frame
# scales roughly with pixel count at a fixed quality.
DEFAULT_RUNGS = [
    Rung(640, 360, 2, 0.45, 0.4),
    Rung(960, 540, 3, 0.50, 1.1),
    Rung(960, 540, 4, 0.55, 1.7),     # start here — conservative but usable
    Rung(1280, 720, 4, 0.60, 2.9),
    Rung(1280, 720, 6, 0.62, 4.6),
]


@dataclass
class _CamState:
    rung: int
    since: float = 0.0               # when the current rung took effect
    last_sent: float = 0.0           # when we last pushed a tune message
    sent_rung: int = -1              # what the phone was last told
    latency_ms: float = 0.0
    delivered_fps: float = 0.0
    clock_ok: bool = False
    over_count: int = 0              # consecutive evaluations above demote_ms
    good_since: float = 0.0          # continuously under promote_ms since
    demotes: int = 0
    promotes: int = 0
    seq: int = 0
    # Highest rung believed to actually work on this link, and when we may next
    # try to exceed it. Without this the controller has no memory: it promotes
    # into a rung the link cannot carry, panics back down, finds the lower rung
    # comfortable, and climbs straight back into the same wall forever.
    ceiling: int = 1 << 30
    probe_after: float = 0.0
    probe_backoff: float = 30.0


class UplinkTuner:
    """One per server. `observe()` per frame (cheap), `decide()` at ~1 Hz."""

    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or {})
        self.enabled = bool(c.get("adaptive", True))
        rungs = c.get("rungs")
        self.rungs = ([Rung(**r) for r in rungs] if rungs else list(DEFAULT_RUNGS))
        self.start_rung = self._clamp(int(c.get("start_rung", 2)))
        self.demote_ms = float(c.get("demote_ms", 900))
        self.panic_ms = float(c.get("panic_ms", 2500))
        self.promote_ms = float(c.get("promote_ms", 450))
        self.promote_dwell_s = float(c.get("promote_dwell_s", 10.0))
        self.demote_dwell_s = float(c.get("demote_dwell_s", 2.0))
        self.budget_mbps = float(c.get("uplink_budget_mbps", 12.0))
        self.resend_s = float(c.get("resend_s", 5.0))
        # How long to leave a rung alone after it proved too fast, and the cap
        # on that interval as repeated attempts keep failing.
        self.probe_backoff_s = float(c.get("probe_backoff_s", 30.0))
        self.probe_backoff_max_s = float(c.get("probe_backoff_max_s", 600.0))
        self._cams: dict[int, _CamState] = {}
        self._last_eval = 0.0
        self._last_promote = 0.0
        self._last_demote = 0.0

    def _clamp(self, i: int) -> int:
        return max(0, min(i, len(self.rungs) - 1))

    def _st(self, cam_id: int, now: float) -> _CamState:
        st = self._cams.get(cam_id)
        if st is None:
            st = self._cams[cam_id] = _CamState(rung=self.start_rung, since=now,
                                                good_since=now)
        return st

    def observe(self, cam_id: int, latency_ms: float, delivered_fps: float,
                clock_ok: bool, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        st = self._st(cam_id, now)
        st.latency_ms = float(latency_ms)
        st.delivered_fps = float(delivered_fps)
        st.clock_ok = bool(clock_ok)

    def forget(self, cam_id: int) -> None:
        self._cams.pop(cam_id, None)

    def rung(self, cam_id: int) -> Rung:
        st = self._cams.get(cam_id)
        return self.rungs[st.rung if st else self.start_rung]

    def _msg(self, cam_id: int, st: _CamState) -> dict:
        r = self.rungs[st.rung]
        st.seq += 1
        return {"type": "tune", "seq": st.seq, "w": r.w, "h": r.h,
                "fps": r.fps, "q": r.q, "rung": st.rung, "label": r.label}

    def decide(self, now: float | None = None) -> dict[int, dict]:
        """Returns {cam_id: tune message} for cameras that need telling.

        Evaluated at most once per second. Messages carry absolute state, not a
        delta, so a lost one is harmless — the periodic resend re-asserts it and
        a phone that missed one simply keeps its previous setting meanwhile.
        """
        now = time.monotonic() if now is None else now
        out: dict[int, dict] = {}
        if not self.enabled or not self._cams:
            return out
        if now - self._last_eval < 1.0:
            return self._resends(now)
        self._last_eval = now

        # ── Demotion: unconditional, but only the worst offender moves ───────
        # Every camera shares one bottleneck and so sees much the same latency.
        # Demoting all of them together would overshoot badly and then promote
        # them all together again — the classic synchronised-backoff oscillation.
        worst_id, worst = None, -1.0
        for cam_id, st in self._cams.items():
            if not st.clock_ok:
                continue                       # an untrustworthy clock cannot steer
            if st.latency_ms > self.panic_ms and st.rung > 0:
                if st.latency_ms > worst:
                    worst_id, worst = cam_id, st.latency_ms
                continue
            if st.latency_ms > self.demote_ms:
                st.over_count += 1
                if st.over_count >= 2 and st.latency_ms > worst:
                    worst_id, worst = cam_id, st.latency_ms
            else:
                st.over_count = 0

        # The shared link is over budget on its own — someone must come down
        # even if nobody's latency has blown up yet. Take it off the greediest
        # camera, which is the one contributing most to the shared cost.
        if worst_id is None and self._committed() > self.budget_mbps:
            over = [(st.rung, cid) for cid, st in self._cams.items() if st.rung > 0]
            if over:
                worst_id = max(over)[1]

        if worst_id is not None:
            st = self._cams[worst_id]
            # One camera at a time, globally: after each step the link needs a
            # moment to actually respond. Demoting everyone before that shows up
            # overshoots, and then they all promote back together.
            if (now - st.since >= self.demote_dwell_s
                    and now - self._last_demote >= self.demote_dwell_s
                    and st.rung > 0):
                # Panic drops two rungs at once: at that latency one step at a
                # time takes too long to matter.
                step = 2 if st.latency_ms > self.panic_ms else 1
                st.rung = self._clamp(st.rung - step)
                # Remember what failed, and wait longer each time it fails again.
                st.ceiling = st.rung
                st.probe_after = now + st.probe_backoff
                st.probe_backoff = min(st.probe_backoff * 2, self.probe_backoff_max_s)
                st.since = st.good_since = now
                st.over_count = 0
                st.demotes += 1
                self._last_demote = now
                logger.info("CAM-%02d uplink -> %s (latency %.0f ms)",
                            worst_id, self.rungs[st.rung].label, st.latency_ms)
                out[worst_id] = self._msg(worst_id, st)

        # ── Promotion: slow, one camera at a time, inside a shared budget ────
        if not out:
            for cam_id, st in sorted(self._cams.items()):
                if st.rung >= len(self.rungs) - 1 or not st.clock_ok:
                    continue
                if st.latency_ms >= self.promote_ms:
                    st.good_since = now
                    continue
                if now - st.good_since < self.promote_dwell_s:
                    continue
                # A rung that already failed stays off-limits until its probe
                # timer expires; the timer doubles each time, so a link that
                # genuinely cannot carry it is left alone rather than retried
                # forever. The lower rung feeling comfortable is not evidence.
                if st.rung >= st.ceiling and now < st.probe_after:
                    continue
                # Only promote a phone that is actually honouring its rung —
                # otherwise we would "reward" a client that ignored the tune
                # (an old cached page) and make the shared link worse.
                want = self.rungs[st.rung].fps * 0.85
                if st.delivered_fps < want:
                    continue
                if now - self._last_promote < self.promote_dwell_s:
                    break                       # one promotion at a time, globally
                nxt = self.rungs[st.rung + 1]
                if self._committed() - self.rungs[st.rung].mbps + nxt.mbps > self.budget_mbps:
                    continue
                st.rung += 1
                st.since = st.good_since = now
                st.promotes += 1
                self._last_promote = now
                logger.info("CAM-%02d uplink -> %s (latency %.0f ms)",
                            cam_id, nxt.label, st.latency_ms)
                out[cam_id] = self._msg(cam_id, st)
                break

        for cam_id, msg in out.items():
            st = self._cams[cam_id]
            st.last_sent, st.sent_rung = now, st.rung
        out.update(self._resends(now))
        return out

    def _committed(self) -> float:
        """Total bitrate we have asked for. A phone we have never successfully
        tuned is assumed to be running unthrottled, so it still counts against
        the budget and the others stay protected from it."""
        total = 0.0
        for st in self._cams.values():
            total += self.rungs[st.rung].mbps if st.sent_rung == st.rung else 8.0
        return total

    def _resends(self, now: float) -> dict[int, dict]:
        """Re-assert current state periodically: a phone that connected before
        the tuner had an opinion, or missed a message, converges on its own."""
        out = {}
        for cam_id, st in self._cams.items():
            if st.sent_rung != st.rung or now - st.last_sent >= self.resend_s:
                out[cam_id] = self._msg(cam_id, st)
                st.last_sent, st.sent_rung = now, st.rung
        return out

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "budget_mbps": self.budget_mbps,
            "committed_mbps": round(self._committed(), 1),
            "cameras": {
                str(cam_id): {
                    "rung": st.rung, "label": self.rungs[st.rung].label,
                    "fps": self.rungs[st.rung].fps, "q": self.rungs[st.rung].q,
                    "latency_ms": round(st.latency_ms),
                    "delivered_fps": round(st.delivered_fps, 1),
                    "clock_ok": st.clock_ok,
                    "demotes": st.demotes, "promotes": st.promotes,
                }
                for cam_id, st in sorted(self._cams.items())
            },
        }
