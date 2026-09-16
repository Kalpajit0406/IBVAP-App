"""
face_events.py — event-triggered face recognition: "when a person is detected,
capture up to 5 face snapshots, decide who (if anyone) they are, then go idle"
instead of running face detection+embedding on every tracked person every tick.

Two pieces, mirroring anpr_events.py's split:

  * `PersonArrivalTracker` — a pure, GPU-free per-(camera, track) burst
    scheduler + majority-vote decider. The first tick a person track is seen
    opens a capture "burst" budget: up to `burst_n` face-bearing readings (a
    tick where FaceEngine found no usable face in the crop doesn't count
    against the burst, only against the attempt/time budget). Once burst_n
    readings are collected — or the attempt/time budget runs out first — the
    burst finalizes into one `FaceVerdict`: the identity (if any) that at least
    `vote_min` of the (<=burst_n) readings agreed on, each individually past
    FaceEngine's match_threshold. A single lucky/adversarial frame is never
    enough on its own — this is face's counterpart to WeaponDetector's
    hold_hits/hold_window temporal vote, since a confirmed match here drives a
    Critical override in risk_engine.py.

  * `FaceResultWriter` — writes the *separate* face-results output folder
    (deliberately not a 4th SnapshotWriter reason, same reasoning as ANPR): one
    small burst of pictures + one JSON record per person arrival, under
    data/face_results/<cam_id>/, carrying the matched identity (or "unknown"),
    similarity, and vote count together. Each burst-tick picture is written
    immediately via capture_snapshot() (so it exists even if the burst never
    cleanly finalizes) and the record is completed later by finalize() — or, if
    a burst never finalizes (e.g. the pipeline hot-swaps mid-burst), by
    sweep_timeouts() so the pictures collected so far are still filed.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .imaging import encode_jpeg, slug

logger = logging.getLogger("ibvap.face_events")

REASON = "face"


# ── burst / vote state machine ──────────────────────────────────────────────

@dataclass
class FaceVerdict:
    cam_id: int
    track_id: int
    ts: float
    matched_id: Optional[str]
    matched_name: Optional[str]
    similarity: float          # best similarity among the agreeing votes; 0.0 if none
    votes: int                 # how many of the collected readings agreed on matched_id
    of: int                    # how many face-bearing readings were collected (<= burst_n)
    on_watchlist: bool


@dataclass
class _FaceTrackState:
    first_tick: int
    first_ts: float
    attempts: int = 0                         # capture ticks offered, whether or not a face was found
    hits: list = field(default_factory=list)  # FaceHit list, only entries where a face WAS found
    last_watchlist_hit: object = None         # most recent on_watchlist FaceHit, for the TTL cache
    last_watchlist_ts: float = 0.0
    done: bool = False


class PersonArrivalTracker:
    """Per-(cam_id, track_id) burst scheduler + majority-vote decider. No GPU, no I/O."""

    def __init__(self, burst_n: int = 5, max_attempts: int = 15,
                 attempt_window_s: float = 8.0, vote_min: int = 3,
                 hit_ttl_s: float = 4.0) -> None:
        self._burst_n = max(1, int(burst_n))
        self._max_attempts = max(self._burst_n, int(max_attempts))
        self._window_s = max(0.0, float(attempt_window_s))
        self._vote_min = max(1, min(int(vote_min), self._burst_n))
        self._ttl = float(hit_ttl_s)
        self._state: dict[tuple[int, int], _FaceTrackState] = {}

    def wants_capture(self, key: tuple[int, int], tick: int, now: float) -> bool:
        """True while this person track still needs more burst readings and is
        within its attempt/time budget. Opens the track's state on first call."""
        st = self._state.get(key)
        if st is None:
            self._state[key] = _FaceTrackState(first_tick=tick, first_ts=now)
            return True
        if st.done:
            return False
        return (len(st.hits) < self._burst_n
                and st.attempts < self._max_attempts
                and (now - st.first_ts) <= self._window_s)

    def attempts_so_far(self, key: tuple[int, int]) -> int:
        st = self._state.get(key)
        return st.attempts if st else 0

    def record(self, key: tuple[int, int], now: float, hit) -> Optional[FaceVerdict]:
        """Call once per tick a capture was attempted for `key` (`hit` is the
        FaceHit found in that crop, or None if no usable face was found this
        tick). Returns a FaceVerdict the tick the burst finalizes (burst_n
        face-bearing hits collected, or the attempt/time budget exhausted),
        else None. Also called with a fresh key by _decide's caller isn't
        required — wants_capture() already opens the state."""
        st = self._state.setdefault(key, _FaceTrackState(first_tick=0, first_ts=now))
        st.attempts += 1
        if hit is not None:
            st.hits.append(hit)
            if getattr(hit, "on_watchlist", False):
                st.last_watchlist_hit, st.last_watchlist_ts = hit, now
        finished = (len(st.hits) >= self._burst_n
                    or st.attempts >= self._max_attempts
                    or (now - st.first_ts) > self._window_s)
        if not finished:
            return None
        st.done = True
        return self._decide(key, st, now)

    def _decide(self, key: tuple[int, int], st: _FaceTrackState, now: float) -> FaceVerdict:
        cam_id, track_id = key
        of = len(st.hits)
        if of == 0:
            return FaceVerdict(cam_id, track_id, now, None, None, 0.0, 0, 0, False)
        tally: dict[Optional[str], list] = {}
        for h in st.hits:
            gid = h.matched_id if h.on_watchlist else None   # only count confident votes
            tally.setdefault(gid, []).append(h)
        best_id, best_hits = max(tally.items(), key=lambda kv: len(kv[1]))
        votes = len(best_hits)
        if best_id is not None and votes >= self._vote_min:
            name = best_hits[0].matched_name
            sim = max(h.similarity for h in best_hits)
            return FaceVerdict(cam_id, track_id, now, best_id, name,
                               round(sim, 3), votes, of, True)
        return FaceVerdict(cam_id, track_id, now, None, None, 0.0, votes, of, False)

    def fresh_hit(self, key: tuple[int, int], now: float):
        """Most recent on_watchlist FaceHit for `key` if still within
        hit_ttl_s — keeps the box/banner alive between bursts, mirrors
        WeaponDetector._fresh_by_cam / hit_ttl_s."""
        st = self._state.get(key)
        if st is None or st.last_watchlist_hit is None:
            return None
        return st.last_watchlist_hit if now - st.last_watchlist_ts <= self._ttl else None

    def flush(self, key: tuple[int, int]) -> None:
        """Forget a dead track — a later re-appearance under the same key
        (should not normally happen once ByteTrack drops it) is treated as a
        fresh arrival."""
        self._state.pop(key, None)

    def prune(self, cam_id: int, live_track_ids) -> None:
        """Drop state for every tracked person on this camera that is no
        longer live. ByteTrack ids are not reused, so — unlike
        SnapshotWriter's cooldown-guarded prune — there is no flicker risk in
        dropping immediately."""
        live = {int(t) for t in live_track_ids}
        for k in [k for k in self._state if k[0] == cam_id and k[1] not in live]:
            self._state.pop(k, None)

    def status(self) -> dict:
        return {"tracked": len(self._state),
                "done": sum(1 for s in self._state.values() if s.done)}


# ── dedicated output folder ─────────────────────────────────────────────────

class FaceResultWriter:
    """Per-camera face-result store: a small burst of JPEGs + one index.jsonl
    line per person arrival, separate from data/snapshots/ (breach/weapon/
    posture) and from data/anpr_results/ — a face arrival's own metadata shape
    (matched identity, similarity, vote count, a multi-file burst instead of
    one picture) doesn't fit either of those record shapes."""

    def __init__(self, config: dict) -> None:
        fc = (config or {}).get("face_results", {}) or {}
        self.enabled = bool(fc.get("enabled", True))
        self._dir = Path(fc.get("dir", "data/face_results"))
        self._jpeg_q = int(fc.get("jpeg_quality", 90))
        self.log_to_evidence = bool(fc.get("log_to_evidence", True))

        # (cam,tid) -> partial record: {ts, cam_id, track_id, files: [...], sha256s: [...]}
        self._pending: dict[tuple[int, int], dict] = {}
        self._cam_dirs: set[int] = set()
        self._written = 0

        if self.enabled:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning("face_results disabled — cannot create %s: %s", self._dir, e)
                self.enabled = False

        if self.enabled:
            logger.info("Face results ready — dir=%s  jpeg_q=%d", self._dir, self._jpeg_q)
        else:
            logger.info("Face results disabled")

    def _ensure_cam(self, cam_id: int) -> None:
        if cam_id in self._cam_dirs:
            return
        try:
            (self._dir / str(cam_id)).mkdir(parents=True, exist_ok=True)
            self._cam_dirs.add(cam_id)
        except OSError as e:
            logger.debug("face_results mkdir %s failed: %s", cam_id, e)

    def capture_snapshot(self, cam_id: int, frame, track_id: int, seq: int,
                          meta: dict | None = None,
                          now: float | None = None) -> dict | None:
        """Writes ONE burst-tick face crop immediately (`seq` = 0-based index
        within this arrival's burst) and appends it to the arrival's pending
        record, opening one if this is the first tick of the burst."""
        if not self.enabled:
            return None
        meta = meta or {}
        now = time.time() if now is None else now
        cam_id, track_id = int(cam_id), int(track_id)
        enc = encode_jpeg(frame, self._jpeg_q)
        if enc is None:
            return None
        data, sha = enc
        try:
            self._ensure_cam(cam_id)
            lt = time.localtime(now)
            stamp = time.strftime("%Y%m%d_%H%M%S", lt) + f"_{int((now % 1) * 1000):03d}"
            stem = f"{stamp}_{REASON}_track{track_id}_{seq}"
            detail = slug(meta.get("detail", ""))
            if detail:
                stem += f"_{detail}"
            (self._dir / str(cam_id) / f"{stem}.jpg").write_bytes(data)
        except OSError as e:
            logger.debug("face_results write failed (cam %s, track %s): %s",
                        cam_id, track_id, e)
            return None

        key = (cam_id, track_id)
        rec = self._pending.setdefault(key, {
            "ts": round(now, 3), "cam_id": cam_id, "track_id": track_id,
            "files": [], "sha256s": [],
        })
        file_ref = f"{cam_id}/{stem}.jpg"
        rec["files"].append(file_ref)
        rec["sha256s"].append(sha)
        return {"file": file_ref, "sha256": sha}

    def finalize(self, cam_id: int, track_id: int, verdict: "object",
                 now: float | None = None) -> dict | None:
        """Pops the pending burst for (cam_id, track_id), merges the verdict
        (matched_id/name/similarity/votes/of/on_watchlist), returns the
        completed record — NOT yet written to index.jsonl. The caller attaches
        ``ev_hash`` (from the evidence chain) first and passes it to
        :meth:`record`, the same two-step handoff AnprResultWriter uses.
        Returns None if nothing was pending (e.g. the burst never found a
        single usable face — capture_snapshot was never called)."""
        key = (int(cam_id), int(track_id))
        rec = self._pending.pop(key, None)
        if rec is None:
            return None
        rec.update(
            matched_id=getattr(verdict, "matched_id", None),
            matched_name=getattr(verdict, "matched_name", None),
            similarity=round(float(getattr(verdict, "similarity", 0.0)), 3),
            votes=int(getattr(verdict, "votes", 0)),
            of=int(getattr(verdict, "of", 0)),
            on_watchlist=bool(getattr(verdict, "on_watchlist", False)),
            ev_hash=None,
        )
        return rec

    def sweep_timeouts(self, now: float | None = None, timeout_s: float = 20.0) -> list[dict]:
        """Safety net: complete (as "unknown", 0 votes) any pending burst
        abandoned mid-way (e.g. a hot-swap or track loss raced the finalize
        call) so its pictures aren't silently orphaned. Like :meth:`finalize`,
        returned records are not yet written; the caller attaches ``ev_hash``
        and passes each to :meth:`record`."""
        if not self.enabled or not self._pending:
            return []
        now = time.time() if now is None else now
        stale = [k for k, r in self._pending.items() if now - r["ts"] > timeout_s]
        out = []
        for k in stale:
            rec = self._pending.pop(k)
            rec.update(matched_id=None, matched_name=None, similarity=0.0,
                      votes=0, of=len(rec.get("files", [])), on_watchlist=False,
                      ev_hash=None)
            out.append(rec)
        return out

    def record(self, rec: dict) -> None:
        """Append the completed record (with its ``ev_hash``) to the camera's
        ``index.jsonl`` and bump the counters."""
        if not self.enabled or not rec:
            return
        try:
            idx = self._dir / str(int(rec["cam_id"])) / "index.jsonl"
            with idx.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except (OSError, KeyError, ValueError) as e:
            logger.debug("face_results index append failed: %s", e)
            return
        self._written += 1

    def recent(self, cam_id: int, limit: int = 20) -> list[dict]:
        idx = self._dir / str(int(cam_id)) / "index.jsonl"
        if not idx.exists():
            return []
        rows = []
        try:
            for ln in idx.read_text("utf-8").splitlines()[-max(1, limit):]:
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    pass
        except OSError:
            return []
        rows.reverse()
        return rows

    def status(self) -> dict:
        return {"enabled": self.enabled, "dir": str(self._dir),
                "written": self._written, "pending": len(self._pending)}
