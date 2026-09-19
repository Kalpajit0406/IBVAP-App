"""
snapshots.py — intrusion / malicious-movement evidence snapshots.

When the inference worker sees one of

  * a virtual-fence breach   (ibvap/geofence.py Breach)
  * a confirmed weapon       (ibvap/weapon.py WeaponEvent)
  * a malicious posture      (crouching / lying — ibvap/posture.py PostureFlags)

it saves a JPEG of that exact frame under ``data/snapshots/<cam_id>/`` — folder
``0`` for camera 0, ``1`` for camera 1, and so on, each created the first time
that camera produces a frame.  Both an *annotated* frame (detector boxes,
skeletons, the fence overlay) and a *raw* frame are written, one ``index.jsonl``
line per event records the pair plus their SHA-256s, and the sha of the
annotated image is handed back to the caller so it can be embedded in the
hash-chained evidence log (ibvap/evidence.py).

These are rare, edge-triggered events — a breach fires once per crossing, and
postures / weapons are debounced per track by ``cooldown_s`` — so, unlike the
continuous-learning Harvester, the JPEG encode + write run inline on the worker
thread.  One 720p frame is ~5-15 ms, well inside the detection budget, and it
keeps the sha available in time to bind it into the evidence record.

Nothing here forces a risk level or touches the pipeline: a disabled or failing
writer is a silent no-op, exactly like the rest of the engine.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .imaging import encode_jpeg, slug as _slug

logger = logging.getLogger("ibvap.snapshots")

REASONS = ("breach", "weapon", "posture", "loiter", "running", "group", "following")


def malicious_postures(detections) -> list:
    """Person detections whose posture is `lying` or `crouching`.

    `chest_aim` / `arms_up` are deliberately excluded — those already route
    through the weapon / risk path.  Kept here (not inline in server.py) so the
    trigger selection is unit-testable.
    """
    out = []
    for d in detections or []:
        if not getattr(d, "is_person", False):
            continue
        p = getattr(d, "posture", None)
        if p is not None and (getattr(p, "lying", False) or getattr(p, "crouching", False)):
            out.append(d)
    return out


class SnapshotWriter:
    """Per-camera evidence-snapshot store.  Owned by the single inference-worker
    thread, so no locking is needed."""

    def __init__(self, config: dict) -> None:
        sc = (config or {}).get("snapshots", {}) or {}
        self.enabled = bool(sc.get("enabled", True))
        self._dir = Path(sc.get("dir", "data/snapshots"))
        self._annotated = bool(sc.get("annotated", True))
        self._raw = bool(sc.get("raw", True))
        self._jpeg_q = int(sc.get("jpeg_quality", 90))
        self._cooldown = float(sc.get("cooldown_s", 20.0))
        self._on = {r: bool(sc.get(f"on_{r}", True)) for r in REASONS}
        # Also force the camera's risk level to Critical on a malicious posture.
        self.escalate_posture = bool(sc.get("escalate_posture", False))

        self._last: dict[tuple, float] = {}     # (cam_id, track_id, reason) -> ts
        # (cam_id, reason) -> (frame_sha, {file, sha256, raw_file, raw_sha256}).
        # Several intruders crossing in one detection pass share one frame — we
        # write the JPEG once and give each their own index line + evidence
        # record pointing at it.
        self._last_frame_sha: dict[tuple, tuple] = {}
        self._cam_dirs: set[int] = set()
        self._counts = {r: 0 for r in REASONS}
        self._written = 0

        if self.enabled:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:                 # read-only fs, bad path, …
                logger.warning("snapshots disabled — cannot create %s: %s", self._dir, e)
                self.enabled = False

        if self.enabled:
            logger.info("Snapshots ready — dir=%s  annotated=%s raw=%s  cooldown=%.0fs  on=%s",
                        self._dir, self._annotated, self._raw, self._cooldown,
                        ",".join(r for r, v in self._on.items() if v) or "-")
        else:
            logger.info("Snapshots disabled")

    # ── folders ───────────────────────────────────────────────────────────
    def ensure_cam(self, cam_id: int) -> None:
        cam_id = int(cam_id)
        if not self.enabled or cam_id in self._cam_dirs:
            return
        try:
            (self._dir / str(cam_id)).mkdir(parents=True, exist_ok=True)
            self._cam_dirs.add(cam_id)
        except OSError as e:
            logger.debug("snapshot mkdir %s failed: %s", cam_id, e)

    # ── debounce ──────────────────────────────────────────────────────────
    def wants(self, reason: str) -> bool:
        return self.enabled and self._on.get(reason, False)

    def should_capture(self, cam_id: int, track_id: int, reason: str,
                       now: float | None = None) -> bool:
        """True at most once per `cooldown_s` for a given (camera, track,
        reason).  Records the timestamp when it returns True."""
        if not self.wants(reason):
            return False
        now = time.time() if now is None else now
        key = (int(cam_id), int(track_id), reason)
        if now - self._last.get(key, 0.0) < self._cooldown:
            return False
        self._last[key] = now
        return True

    def flush_track(self, cam_id: int, track_id: int) -> None:
        self._last.pop((int(cam_id), int(track_id)), None)
        for r in REASONS:
            self._last.pop((int(cam_id), int(track_id), r), None)

    def prune(self, cam_id: int, live_track_ids, now: float | None = None) -> None:
        """Forget debounce state for tracks that are gone *and* whose cooldown
        has already elapsed. Keeping a recent guard alive through a one-pass
        ByteTrack drop-out is deliberate — a flickering track must not re-fire a
        snapshot every time it reappears."""
        cam_id = int(cam_id)
        now = time.time() if now is None else now
        live = {int(t) for t in live_track_ids}
        for k in [k for k in self._last
                  if k[0] == cam_id and k[1] >= 0 and k[1] not in live
                  and now - self._last[k] > self._cooldown]:
            self._last.pop(k, None)

    # ── write ─────────────────────────────────────────────────────────────
    def _encode(self, frame) -> tuple[bytes, str] | None:
        return encode_jpeg(frame, self._jpeg_q)

    def capture(self, cam_id: int, annotated_frame, clean_frame, reason: str,
                meta: dict | None = None, now: float | None = None) -> dict | None:
        """Write the JPEG(s) for one event and return a record dict.  The caller
        adds ``ev_hash`` (the evidence-chain hash of the record that embeds this
        snapshot's ``sha256``) and passes it back to :meth:`record`.

        Returns ``None`` if disabled or nothing could be written.
        """
        if not self.enabled:
            return None
        meta = meta or {}
        now = time.time() if now is None else now
        cam_id = int(cam_id)
        try:
            self.ensure_cam(cam_id)
            cam_dir = self._dir / str(cam_id)
            lt = time.localtime(now)
            stamp = time.strftime("%Y%m%d_%H%M%S", lt) + f"_{int((now % 1) * 1000):03d}"
            track_id = int(meta.get("track_id", -1))
            detail = _slug(meta.get("detail", ""))
            stem = f"{stamp}_{reason}_track{track_id}" + (f"_{detail}" if detail else "")

            rec = {"ts": round(now, 3), "reason": reason, "cam_id": cam_id,
                   "track_id": track_id,
                   "detail": str(meta.get("detail", "")),
                   "file": None, "raw_file": None,
                   "sha256": None, "raw_sha256": None, "deduped": False,
                   "ev_hash": None}

            a_enc = self._encode(annotated_frame) if self._annotated else None
            r_enc = self._encode(clean_frame) if self._raw else None
            if a_enc is None and r_enc is None:
                return None

            key_sha = (a_enc or r_enc)[1]
            prev = self._last_frame_sha.get((cam_id, reason))
            if prev is not None and prev[0] == key_sha:
                p = prev[1]
                rec.update(file=p["file"], sha256=p["sha256"],
                           raw_file=p["raw_file"], raw_sha256=p["raw_sha256"],
                           deduped=True)
                return rec

            if a_enc is not None:
                data, sha = a_enc
                (cam_dir / f"{stem}.jpg").write_bytes(data)
                rec["file"], rec["sha256"] = f"{cam_id}/{stem}.jpg", sha
            if r_enc is not None:
                data, sha = r_enc
                (cam_dir / f"{stem}_raw.jpg").write_bytes(data)
                rec["raw_file"], rec["raw_sha256"] = f"{cam_id}/{stem}_raw.jpg", sha
            self._last_frame_sha[(cam_id, reason)] = (key_sha, {
                "file": rec["file"], "sha256": rec["sha256"],
                "raw_file": rec["raw_file"], "raw_sha256": rec["raw_sha256"]})
            return rec
        except OSError as e:
            logger.debug("snapshot write failed (cam %s, %s): %s", cam_id, reason, e)
            return None

    def record(self, rec: dict) -> None:
        """Append the completed record (with its ``ev_hash``) to the camera's
        ``index.jsonl`` and bump the counters."""
        if not self.enabled or not rec:
            return
        reason = rec.get("reason", "")
        try:
            idx = self._dir / str(int(rec["cam_id"])) / "index.jsonl"
            with idx.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except (OSError, KeyError, ValueError) as e:
            logger.debug("snapshot index append failed: %s", e)
            return
        self._written += 1
        if reason in self._counts:
            self._counts[reason] += 1

    def recent(self, cam_id: int, limit: int = 20) -> list[dict]:
        """Tail of one camera's index.jsonl (newest first)."""
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
                "written": self._written, "by_reason": dict(self._counts),
                "escalate_posture": self.escalate_posture}
