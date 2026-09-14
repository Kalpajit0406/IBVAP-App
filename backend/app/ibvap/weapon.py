"""
weapon.py — trained firearm detector, wired into the IBVAP pipeline.

A small dedicated YOLO model (`models/weapon_detector.pt`, one class: `gun`,
trained by `train_weapon.py` on YouTube-GDD) runs as one extra batched pass over
every camera's frame — the same shape as the pose / ANPR passes. It does **not**
touch the person/vehicle detector or its mAP.

This is the object-level counterpart to `src/posture.py` Rule 5 (`chest_aim` /
"AIM"), which is a 2-D skeleton heuristic with no view of any weapon. The two are
fused in `src/risk_engine.py`:

  * AIM posture only          → Critical, "WEAPON (posture)"
  * confirmed gun only        → Critical, "GUN DETECTED"
  * AIM + confirmed gun, same track → Critical score 99, "ARMED THREAT"

"Confirmed" keeps a false Critical rare without demanding a perfect streak:

  1. conf >= `weapon.min_conf` (default 0.30) to count as a candidate at all
  2. box area >= `weapon.min_box_area` px^2
  3. the box is *held* — at least `weapon.person_contain_min` of it lies inside
     a tracked person's box (grown by 10%), and it is smaller than
     `weapon.max_box_person_frac` of that person. Merely touching a person's box
     is not enough: background objects behind a walker (parked motorbikes, a
     stair rail) overlapping the top of the box were being "held". When the
     pose model sees the person's wrists, one of them must also be at the gun
     box (a held gun is in a hand).
  4. temporal vote: the same person track carried a held gun in at least
     `weapon.hold_hits` of its last `weapon.hold_window` weapon passes, and one
     of those hits reached `weapon.confirm_conf`; OR
  5. fusion: the holding person is in an AIM posture (posture.py Rule 5) —
     skeleton and object model agreeing is itself strong evidence.

The old rule (N *consecutive* passes at conf >= 0.45) threw most real guns
away: the model's per-pass recall on held guns is ~0.73, so three hits in a
row happens ~39% of the time, while 2 of the last 4 happens ~94% of the time.
Once confirmed, a track stays confirmed while it has any hit in the window, so
the banner doesn't strobe on a single missed pass.

A hit that fails 3-5 is returned but marked `confirmed=False` — drawn faint,
never alerted.

If the weights or `ultralytics` are missing the engine disables itself with one
warning and the rest of the pipeline is unaffected (same as `AnprEngine`), so
this module can ship and run before `train_weapon.py` has produced the model.
"""
from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("ibvap.weapon")


@dataclass
class WeaponHit:
    bbox: tuple[int, int, int, int]      # gun box, full-frame pixels
    conf: float
    cls_name: str
    person_track: int                     # nearest/holding person track id, or -1
    confirmed: bool                       # passed association + temporal vote


@dataclass
class WeaponEvent:
    cam_id: int
    track_id: int
    conf: float
    bbox: tuple[int, int, int, int]
    ts: float
    tier: str = "gun"                     # "gun" (confirmed) — "armed" is decided in risk_engine


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _centre_in(box, outer, grow: float = 0.15) -> bool:
    """True if the centre of `box` falls inside `outer` grown by `grow` on each side."""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    ow = outer[2] - outer[0]
    oh = outer[3] - outer[1]
    return (outer[0] - ow * grow <= cx <= outer[2] + ow * grow and
            outer[1] - oh * grow <= cy <= outer[3] + oh * grow)


class WeaponDetector:
    """One batched firearm pass across every camera in a detection batch."""

    def __init__(self, cfg: dict, device: str = "0", max_batch: int = 8) -> None:
        self.available = False
        self._cfg = cfg or {}
        self._device = str(device)
        self._min_conf = float(self._cfg.get("min_conf", 0.30))
        self._confirm_conf = max(self._min_conf, float(self._cfg.get("confirm_conf", 0.40)))
        self._min_area = int(self._cfg.get("min_box_area", 400))
        # On real CCTV the model's false positives are person-sized or larger
        # boxes (it latches onto a whole dark silhouette). A held gun is always
        # a fraction of its holder, and never a fifth of a surveillance frame.
        self._max_frame_frac = float(self._cfg.get("max_box_frame_frac", 0.20))
        self._max_person_frac = float(self._cfg.get("max_box_person_frac", 0.55))
        self._person_iou = float(self._cfg.get("person_iou_min", 0.05))
        self._contain = float(self._cfg.get("person_contain_min", 0.5))
        self._wrist_margin = float(self._cfg.get("wrist_margin", 0.12))
        # `hold_frames` is the old consecutive-pass key; honour it if that is
        # all an older config sets.
        self._hold = max(1, int(self._cfg.get("hold_hits", self._cfg.get("hold_frames", 2))))
        self._window = max(self._hold, int(self._cfg.get("hold_window", 4)))
        self._fuse_aim = bool(self._cfg.get("confirm_on_aim", True))
        self._detect_every = max(1, int(self._cfg.get("detect_every", 2)))
        self._imgsz = int(self._cfg.get("imgsz", 640))
        self._ttl = float(self._cfg.get("hit_ttl_s", 1.5))
        self._max_batch = max(1, int(max_batch))

        # (cam_id, person_track) -> deque[(pass_no, conf)] of held-gun hits
        self._history: dict[tuple, collections.deque] = {}
        self._pass = 0
        # (cam_id, person_track) -> (WeaponHit, ts) — most recent, for a short TTL
        self._hits: dict[tuple, tuple] = {}
        # (cam_id, person_track) -> last emitted tier, so we only log transitions
        self._emitted: dict[tuple, str] = {}
        self._events: list[WeaponEvent] = []
        self._n_raw = 0
        self._n_confirmed = 0

        weights = str(self._cfg.get("weights", "models/weapon_detector.pt"))
        if not Path(weights).exists():
            logger.warning("Weapon detector disabled — weights not found at %s "
                           "(train it: python training/train_weapon.py). See docs/WEAPON_DETECTION.md",
                           weights)
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:                          # pragma: no cover
            logger.warning("Weapon detector disabled — %s", e)
            return
        try:
            self._model = YOLO(weights)
            blank = np.zeros((self._imgsz, self._imgsz, 3), np.uint8)
            for n in sorted({1, self._max_batch}):
                self._model.predict([blank] * n, imgsz=self._imgsz,
                                    device=self._device, verbose=False)
        except Exception as e:                            # pragma: no cover
            logger.error("Weapon detector disabled — failed to initialise (%s)", e)
            return

        self._names = getattr(self._model, "names", {0: "gun"}) or {0: "gun"}
        self.available = True
        logger.info("Weapon detector ready — %s, conf>=%.2f (confirm %.2f), %d of %d passes, "
                    "every %d ticks, imgsz %d", Path(weights).name, self._min_conf,
                    self._confirm_conf, self._hold, self._window, self._detect_every, self._imgsz)

    # ── one batched pass (inference-worker thread, like _pose_pass) ──────────
    def begin_pass(self) -> None:
        """Start a new weapon pass (the unit the temporal vote counts in)."""
        self._pass += 1

    def detect_batch(self, per_cam: list, tick: int) -> dict[int, list[WeaponHit]]:
        """per_cam: list of (cam_id, frame_bgr, persons) where persons is a list
        of (x1, y1, x2, y2, track_id[, aiming[, wrists]]) — `wrists` is a list of
        visible (x, y) wrist points from the pose model, or empty/None when
        unknown. Returns {cam_id: [WeaponHit, ...]}."""
        if not self.available:
            return {}
        if tick % self._detect_every != 0:
            # Not a weapon pass this tick — serve the still-fresh confirmed hits
            # so boxes/banner don't strobe between passes.
            return self._fresh_by_cam()

        frames = [f for _, f, _ in per_cam]
        if not frames:
            return {}
        try:
            outs = self._model.predict(frames, conf=self._min_conf, imgsz=self._imgsz,
                                       device=self._device, verbose=False)
        except Exception as e:                            # pragma: no cover
            logger.debug("weapon predict failed: %s", e)
            return self._fresh_by_cam()

        now = time.time()
        self.begin_pass()
        out_by_cam: dict[int, list[WeaponHit]] = {}

        for (cam_id, _frame, persons), out in zip(per_cam, outs):
            hits: list[WeaponHit] = []
            raw = getattr(out, "boxes", None)
            for i in range(len(raw) if raw is not None else 0):
                x1, y1, x2, y2 = (int(v) for v in raw.xyxy[i].tolist())
                conf = float(raw.conf[i])
                cls = int(raw.cls[i]) if raw.cls is not None else 0
                fh, fw = _frame.shape[:2]
                hit = self._classify_box(cam_id, (x1, y1, x2, y2), conf,
                                         self._names.get(cls, "gun"), persons, now,
                                         frame_area=fh * fw)
                if hit is None:
                    continue
                hits.append(hit)
            out_by_cam[cam_id] = hits

        # A held-gun track with no hit left inside the vote window is cleared.
        self._expire_history()

        # Fold in still-fresh confirmed hits for cams that had none this pass.
        for cam_id, fresh in self._fresh_by_cam().items():
            out_by_cam.setdefault(cam_id, [])
            have = {h.person_track for h in out_by_cam[cam_id]}
            out_by_cam[cam_id].extend(h for h in fresh if h.person_track not in have)
        return out_by_cam

    def _expire_history(self) -> None:
        oldest = self._pass - self._window
        for key in list(self._history):
            hist = self._history[key]
            while hist and hist[0][0] <= oldest:
                hist.popleft()
            if not hist:
                self._history.pop(key, None)
                self._emitted.pop(key, None)

    def _associate(self, gbox, persons) -> tuple[int, float]:
        """Which tracked person is holding `gbox`? → (track_id or -1, score).

        Score = the fraction of the gun box inside the person's box grown by
        10% on each side (arms and muzzles reach just past a detector box)."""
        gx1, gy1, gx2, gy2 = gbox
        g_area = max(1.0, float((gx2 - gx1) * (gy2 - gy1)))
        ptrack, best = -1, 0.0
        for person in persons:
            px1, py1, px2, py2, tid = person[:5]
            gw, gh = (px2 - px1) * 0.10, (py2 - py1) * 0.10
            ix = max(0.0, min(gx2, px2 + gw) - max(gx1, px1 - gw))
            iy = max(0.0, min(gy2, py2 + gh) - max(gy1, py1 - gh))
            score = ix * iy / g_area
            gcx, gcy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
            if not (px1 <= gcx <= px2 and py1 <= gcy <= py2):
                continue                         # centre outside the holder: background
            if score < self._contain or score <= best:
                continue
            wrists = person[6] if len(person) > 6 else None
            if wrists:
                m = (py2 - py1) * self._wrist_margin
                if not any(gx1 - m <= wx <= gx2 + m and gy1 - m <= wy <= gy2 + m
                           for wx, wy in wrists):
                    continue                     # nothing in this person's hands there
            ptrack, best = int(tid), score
        return ptrack, best

    def _classify_box(self, cam_id: int, gbox: tuple, conf: float,
                      cls_name: str, persons: list, now: float,
                      frame_area: int | None = None) -> WeaponHit | None:
        """One raw gun box → a WeaponHit (or None if filtered). Applies the
        area gate, person association and the temporal vote; records the hit +
        emits a WeaponEvent on the first pass it becomes confirmed. Pure enough
        to unit-test without a model."""
        x1, y1, x2, y2 = gbox
        if conf < self._min_conf:
            return None
        area = (x2 - x1) * (y2 - y1)
        if area < self._min_area:
            return None
        if frame_area and area > self._max_frame_frac * frame_area:
            return None
        self._n_raw += 1

        ptrack, _score = self._associate(gbox, persons)
        if ptrack >= 0:
            holder = next(p for p in persons if int(p[4]) == ptrack)
            p_area = max(1, (holder[2] - holder[0]) * (holder[3] - holder[1]))
            if area > self._max_person_frac * p_area:
                return None                    # silhouette-sized "gun": not a weapon
        held = ptrack >= 0
        confirmed = False
        if held:
            key = (cam_id, ptrack)
            hist = self._history.setdefault(key, collections.deque())
            if hist and hist[-1][0] == self._pass:           # 2 boxes, 1 person, 1 pass
                hist[-1] = (self._pass, max(hist[-1][1], conf))
            else:
                hist.append((self._pass, conf))
            oldest = self._pass - self._window
            while hist and hist[0][0] <= oldest:
                hist.popleft()
            aiming = any(len(p) > 5 and p[5] and int(p[4]) == ptrack for p in persons)
            voted = (len(hist) >= self._hold
                     and max(c for _n, c in hist) >= self._confirm_conf)
            confirmed = (voted
                         or (self._fuse_aim and aiming)
                         or self._emitted.get(key) == "gun")   # stays up inside the window

        hit = WeaponHit(gbox, round(conf, 3), cls_name, ptrack, confirmed)
        if held:
            self._hits[key] = (hit, now)
            if confirmed:
                self._n_confirmed += 1
                if self._emitted.get(key) != "gun":
                    self._emitted[key] = "gun"
                    self._events.append(WeaponEvent(
                        cam_id, ptrack, hit.conf, gbox, now, "gun"))
        return hit

    def _fresh_by_cam(self) -> dict[int, list[WeaponHit]]:
        now = time.time()
        out: dict[int, list[WeaponHit]] = {}
        for (cam_id, _tid), (hit, ts) in list(self._hits.items()):
            if now - ts > self._ttl:
                self._hits.pop((cam_id, _tid), None)
                continue
            if hit.confirmed:
                out.setdefault(cam_id, []).append(hit)
        return out

    # ── consumer interface (inference-worker thread) ───────────────────────
    def drain_events(self) -> list[WeaponEvent]:
        out, self._events = self._events, []
        return out

    def flush_track(self, key) -> None:
        """`key` is (cam_id, person_track) — call when that person track is lost."""
        self._history.pop(key, None)
        self._hits.pop(key, None)
        self._emitted.pop(key, None)

    def status(self) -> dict:
        now = time.time()
        active = [
            {"cam": cam_id, "track": tid, "conf": hit.conf,
             "confirmed": hit.confirmed}
            for (cam_id, tid), (hit, ts) in sorted(self._hits.items())
            if now - ts < self._ttl
        ]
        return {
            "enabled": self.available,
            "raw": self._n_raw,
            "confirmed": self._n_confirmed,
            "hold_hits": self._hold,
            "hold_window": self._window,
            "min_conf": self._min_conf,
            "confirm_conf": self._confirm_conf,
            "active": active,
        }

    def stop(self) -> None:
        pass
