"""
anpr_events.py — event-triggered ANPR: "when a vehicle is detected, capture it
once, read its plate + type, then go idle" instead of continuously re-scanning
every tracked vehicle on every qualifying tick.

Two pieces:

  * `VehicleArrivalTracker` — a pure, GPU-free per-(camera, track) state
    machine. The first tick a vehicle track is seen is its "arrival" (fires a
    snapshot + plate submit + vehicle-type classify, once). A bounded number of
    "followup" ticks keep feeding AnprEngine's own multi-read vote() so the
    plate can still improve, then the track goes "done" — genuinely idle, no
    further work submitted for it until it disappears and a new track_id
    arrives (a real re-arrival).

  * `AnprResultWriter` — writes the *separate* ANPR output folder the user
    asked for (deliberately not a 4th SnapshotWriter reason): one picture +
    one JSON record per vehicle arrival, under data/anpr_results/<cam_id>/,
    carrying the plate, the vehicle type, and both confidences together. Plate
    OCR resolves asynchronously (AnprEngine's worker thread) after the
    snapshot + vehicle-type are already known, so a record is opened at
    capture_snapshot() and completed later by finalize() — or, if the plate
    never resolves, by sweep_timeouts() so the type + picture are still filed.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .imaging import encode_jpeg, slug

logger = logging.getLogger("ibvap.anpr_events")

REASON = "arrival"


# ── arrival / idle state machine ────────────────────────────────────────────

@dataclass
class _TrackState:
    first_tick: int
    first_ts: float
    followups: int = 0
    done: bool = False


class VehicleArrivalTracker:
    """Per-(cam_id, track_id) arrival/idle classifier. No GPU, no I/O."""

    def __init__(self, follow_up_max: int = 8, follow_up_s: float = 6.0) -> None:
        self._max = max(0, int(follow_up_max))
        self._window_s = max(0.0, float(follow_up_s))
        self._state: dict[tuple[int, int], _TrackState] = {}

    def classify(self, key: tuple[int, int], tick: int, now: float,
                 confident: bool) -> str:
        """Returns "arrival" (fire once), "followup" (bounded extra reads),
        or "done" (true idle — nothing should be submitted for this track)."""
        st = self._state.get(key)
        if st is None:
            self._state[key] = _TrackState(first_tick=tick, first_ts=now)
            return "arrival"
        if st.done:
            return "done"
        if confident or st.followups >= self._max or (now - st.first_ts) > self._window_s:
            st.done = True
            return "done"
        st.followups += 1
        return "followup"

    def is_new(self, key: tuple[int, int]) -> bool:
        """True until classify() has seen this track once."""
        return key not in self._state

    def flush(self, key: tuple[int, int]) -> None:
        """Forget a dead track — a later re-appearance under the same key
        (should not normally happen once ByteTrack drops it) is treated as a
        fresh arrival."""
        self._state.pop(key, None)

    def prune(self, cam_id: int, live_track_ids) -> None:
        """Drop state for every tracked vehicle on this camera that is no
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

class AnprResultWriter:
    """Per-camera ANPR-result store: one JPEG + one index.jsonl line per
    vehicle arrival, separate from data/snapshots/ (breach/weapon/posture)."""

    def __init__(self, config: dict) -> None:
        ac = (config or {}).get("anpr_results", {}) or {}
        self.enabled = bool(ac.get("enabled", True))
        self._dir = Path(ac.get("dir", "data/anpr_results"))
        self._jpeg_q = int(ac.get("jpeg_quality", 90))
        self._timeout = float(ac.get("finalize_timeout_s", 6.0))
        self.log_to_evidence = bool(ac.get("log_to_evidence", True))

        self._pending: dict[tuple[int, int], dict] = {}   # (cam,tid) -> partial rec
        self._cam_dirs: set[int] = set()
        self._written = 0

        if self.enabled:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning("anpr_results disabled — cannot create %s: %s", self._dir, e)
                self.enabled = False

        if self.enabled:
            logger.info("ANPR results ready — dir=%s  jpeg_q=%d  finalize_timeout=%.0fs",
                        self._dir, self._jpeg_q, self._timeout)
        else:
            logger.info("ANPR results disabled")

    def _ensure_cam(self, cam_id: int) -> None:
        if cam_id in self._cam_dirs:
            return
        try:
            (self._dir / str(cam_id)).mkdir(parents=True, exist_ok=True)
            self._cam_dirs.add(cam_id)
        except OSError as e:
            logger.debug("anpr_results mkdir %s failed: %s", cam_id, e)

    def capture_snapshot(self, cam_id: int, frame, track_id: int,
                          meta: dict | None = None,
                          now: float | None = None) -> dict | None:
        """Write the arrival image immediately (so it exists even if OCR later
        fails or times out) and open a pending record for this vehicle."""
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
            stem = f"{stamp}_{REASON}_track{track_id}"
            detail = slug(meta.get("detail", ""))
            if detail:
                stem += f"_{detail}"
            (self._dir / str(cam_id) / f"{stem}.jpg").write_bytes(data)
        except OSError as e:
            logger.debug("anpr_results write failed (cam %s, track %s): %s",
                        cam_id, track_id, e)
            return None

        rec = {
            "ts": round(now, 3), "cam_id": cam_id, "track_id": track_id,
            "plate": None, "plate_conf": 0.0, "plate_valid": False,
            "vehicle_type": None, "vehicle_type_conf": 0.0,
            "vehicle_type_fallback": True,
            "file": f"{cam_id}/{stem}.jpg", "sha256": sha, "ev_hash": None,
        }
        self._pending[(cam_id, track_id)] = rec
        return rec

    def attach_vehicle_type(self, cam_id: int, track_id: int, vt) -> None:
        """`vt`: a VehicleTypeResult (ibvap/vehicle_type.py), or anything with
        .label/.conf/.fallback. No-op if this arrival's record has already
        been finalized or never existed (e.g. writer disabled)."""
        rec = self._pending.get((int(cam_id), int(track_id)))
        if rec is None or vt is None:
            return
        rec["vehicle_type"] = vt.label
        rec["vehicle_type_conf"] = round(float(vt.conf), 3)
        rec["vehicle_type_fallback"] = bool(vt.fallback)

    def finalize(self, cam_id: int, track_id: int, plate_reading=None,
                 now: float | None = None) -> dict | None:
        """Complete a pending arrival record (with or without a resolved
        plate) and return it — the record is NOT written to index.jsonl yet.
        The caller attaches ``ev_hash`` (the evidence-chain hash of the record
        that embeds this arrival's ``sha256``) and passes it to :meth:`record`,
        the same two-step handoff SnapshotWriter uses. Returns None if there
        was nothing pending for this (cam, track)."""
        key = (int(cam_id), int(track_id))
        rec = self._pending.pop(key, None)
        if rec is None:
            return None
        if plate_reading is not None:
            rec["plate"] = plate_reading.text
            rec["plate_conf"] = round(float(plate_reading.conf), 3)
            rec["plate_valid"] = bool(plate_reading.valid)
        return rec

    def sweep_timeouts(self, now: float | None = None) -> list[dict]:
        """Complete (with plate=null) any pending arrival whose plate never
        resolved within finalize_timeout_s — the picture + vehicle type are
        still worth filing even when the plate is unreadable. Like
        :meth:`finalize`, the returned records are not yet written; the caller
        attaches ``ev_hash`` and passes each to :meth:`record`."""
        if not self.enabled or not self._pending:
            return []
        now = time.time() if now is None else now
        stale = [k for k, r in self._pending.items()
                 if now - r["ts"] > self._timeout]
        return [self._pending.pop(k) for k in stale]

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
            logger.debug("anpr_results index append failed: %s", e)
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
