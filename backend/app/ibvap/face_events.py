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
    (deliberately not a 4th SnapshotWriter reason, same reasoning as ANPR): two
    small crops + one JSON record per person arrival, under
    data/face_results/<cam_id>/, carrying the matched identity (or "unknown"),
    similarity, and vote count together. Every capture tick offers its crop via
    offer_candidate(); only the best-scoring one survives, and flush_best()
    writes it (plus a person-box context crop) once the burst finalizes — or,
    if a burst somehow never finalizes, sweep_timeouts() flushes it anyway.

    Every person offers a candidate, including those where no face could be
    detected at all — a geometric head crop stands in — so nobody passes the
    camera unrecorded. Buffering is what makes that affordable: one arrival
    costs two small crops rather than a full-resolution frame per burst tick.
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
    last_seen: float = 0.0                    # last tick this track appeared in its camera's detections
    done: bool = False


class PersonArrivalTracker:
    """Per-(cam_id, track_id) burst scheduler + majority-vote decider. No GPU, no I/O."""

    def __init__(self, burst_n: int = 5, max_attempts: int = 15,
                 attempt_window_s: float = 8.0, vote_min: int = 3,
                 hit_ttl_s: float = 4.0, prune_grace_s: float = 2.0,
                 sweep_grace_s: float = 1.0) -> None:
        self._burst_n = max(1, int(burst_n))
        self._max_attempts = max(self._burst_n, int(max_attempts))
        self._window_s = max(0.0, float(attempt_window_s))
        self._vote_min = max(1, min(int(vote_min), self._burst_n))
        self._ttl = float(hit_ttl_s)
        self._prune_grace = max(0.0, float(prune_grace_s))
        self._sweep_grace = max(0.0, float(sweep_grace_s))
        self._state: dict[tuple[int, int], _FaceTrackState] = {}

    def wants_capture(self, key: tuple[int, int], tick: int, now: float) -> bool:
        """True while this person track still needs more burst readings and is
        within its attempt/time budget. Opens the track's state on first call."""
        st = self._state.get(key)
        if st is None:
            self._state[key] = _FaceTrackState(first_tick=tick, first_ts=now,
                                               last_seen=now)
            return True
        st.last_seen = now
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
        st.last_seen = now
        # Only a *trusted* reading (det_score >= face.det_score_min) is allowed to
        # vote or count toward burst_n. Low-confidence readings still reach the
        # writer as snapshot candidates, but must not weaken the identity vote.
        if hit is not None and getattr(hit, "trusted", True):
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

    def sweep_expired(self, now: float) -> list[FaceVerdict]:
        """Finalize every burst whose time budget lapsed without `record()`
        being called on the deciding tick.

        `wants_capture()` and `record()`'s `finished` test are exact
        complements on the time window, so the tick the window expires the
        track simply stops being offered and `record()` is never called again —
        leaving the burst permanently un-finalized. It also stops being offered
        whenever the motion gate closes, the camera is removed, or the person
        leaves frame mid-burst. A sweep covers all of those; de-complementing
        the two predicates would only cover the first.

        Returns a real verdict per swept burst — so a burst that collected
        enough agreeing votes still reports `on_watchlist=True` here rather
        than being written out as "unknown" by FaceResultWriter.sweep_timeouts.
        """
        deadline = self._window_s + self._sweep_grace
        out: list[FaceVerdict] = []
        for key, st in list(self._state.items()):
            if st.done or (now - st.first_ts) <= deadline:
                continue
            st.done = True
            out.append(self._decide(key, st, now))
        return out

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

    def prune(self, cam_id: int, live_track_ids, now: float | None = None) -> None:
        """Drop state for tracked people on this camera absent for longer than
        `prune_grace_s`.

        Dropping the instant a track misses one frame loses the whole burst —
        `hits` resets to 0, `first_ts` restarts, and `last_watchlist_hit` is
        destroyed so a confirmed banner flickers. That matters because a
        detection miss is routine on a blurry or laggy stream. ByteTrack ids
        not being reused only rules out state from one track landing on
        another; it says nothing about losing state for a track that continues
        under the same id — and `tracker.lost_buffer_frames` means ByteTrack
        itself keeps a lost track alive for seconds and re-associates it.
        """
        now = time.time() if now is None else now
        live = {int(t) for t in live_track_ids}
        for k, st in list(self._state.items()):
            if k[0] != cam_id:
                continue
            if k[1] in live:
                st.last_seen = now
            elif now - st.last_seen > self._prune_grace:
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

    def offer_candidate(self, cam_id: int, track_id: int, hit,
                        meta: dict | None = None,
                        now: float | None = None) -> bool:
        """Offer this tick's crop as the burst's saved snapshot; keeps it only
        if it beats the best so far. No I/O — a burst writes once, in
        :meth:`flush_best`.

        Called for EVERY person on a capture tick, including those where no
        face was detectable at all (`hit.source == "head_box"`, or `hit` None),
        so a blurry, side-on, shadowed or distant person is still recorded.
        Buffering rather than writing per tick is what makes that affordable:
        the old path wrote one full-resolution frame per tick and kept them all.
        """
        if not self.enabled:
            return False
        now = time.time() if now is None else now
        cam_id, track_id = int(cam_id), int(track_id)
        crop = getattr(hit, "face_crop", None) if hit is not None else None
        quality = float(getattr(hit, "quality", 0.0)) if hit is not None else 0.0
        if crop is None or crop.size == 0:
            return False

        key = (cam_id, track_id)
        best = self._pending.get(key)
        if best is None:
            best = self._pending.setdefault(key, {
                "ts": round(now, 3), "cam_id": cam_id, "track_id": track_id,
                "files": [], "sha256s": [], "candidates_seen": 0,
                "_rank": (-1, -1.0),
            })
        best["candidates_seen"] += 1
        # Rank on (really a face, then quality): an actually-detected face always
        # beats a geometric head-box guess, however well the guess happens to
        # score. A guess only wins when nothing better was ever offered.
        source = str(getattr(hit, "source", "face"))
        rank = (1 if source == "face" else 0, quality)
        if rank <= best["_rank"]:
            return False

        best.update(_rank=rank, _face=crop,
                    _person=getattr(hit, "person_crop", None),
                    _detail=(meta or {}).get("detail", ""),
                    face_quality=quality,
                    face_source=source,
                    face_bbox=[int(v) for v in getattr(hit, "bbox", (0, 0, 0, 0))],
                    face_det_score=round(float(getattr(hit, "det_score", 0.0)), 3))
        return True

    def flush_best(self, cam_id: int, track_id: int, verdict: "object",
                   now: float | None = None) -> dict | None:
        """Write the burst's best face crop (and its person-context crop), merge
        the verdict, and return the completed record — the same two-step handoff
        :meth:`finalize` uses: the caller attaches ``ev_hash`` then calls
        :meth:`record`. Returns None if the burst never had a usable crop."""
        if not self.enabled:
            return None
        now = time.time() if now is None else now
        key = (int(cam_id), int(track_id))
        rec = self._pending.pop(key, None)
        if rec is None:
            return None
        face, person = rec.pop("_face", None), rec.pop("_person", None)
        detail = rec.pop("_detail", "")
        rec.pop("_rank", None)
        if face is None:
            return None

        lt = time.localtime(now)
        stamp = time.strftime("%Y%m%d_%H%M%S", lt) + f"_{int((now % 1) * 1000):03d}"
        base = f"{stamp}_{REASON}_track{int(track_id)}"
        d = slug(detail)
        if d:
            base += f"_{d}"
        # Index 0 is the face, index 1 the person context — server.py's evidence
        # row and the Flutter panel both read files[0], so the face is what they
        # show without either needing to change.
        for img, suffix in ((face, "face"), (person, "person")):
            if img is None:
                continue
            enc = encode_jpeg(img, self._jpeg_q)
            if enc is None:
                continue
            data, sha = enc
            try:
                self._ensure_cam(int(cam_id))
                stem = f"{base}_{suffix}"
                (self._dir / str(int(cam_id)) / f"{stem}.jpg").write_bytes(data)
            except OSError as e:
                logger.debug("face_results write failed (cam %s, track %s): %s",
                             cam_id, track_id, e)
                continue
            rec["files"].append(f"{int(cam_id)}/{stem}.jpg")
            rec["sha256s"].append(sha)

        if not rec["files"]:
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
        # PersonArrivalTracker.sweep_expired should finalize every burst with a
        # real verdict long before this fires; route through flush_best anyway
        # so a swept burst still writes its best crop rather than a record whose
        # files list is empty and whose buffered images leak into the JSON.
        unknown = FaceVerdict(0, 0, now, None, None, 0.0, 0, 0, False)
        for cam_id, track_id in stale:
            rec = self.flush_best(cam_id, track_id, unknown, now)
            if rec is not None:
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
