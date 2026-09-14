"""
posture.py -- skeleton-based anomaly classifier.

Two classes:
  PoseEstimator   -- thin wrapper around yolo26n-pose.pt; runs on person crops
  PoseClassifier  -- pure-Python geometry rules; no GPU, no model, fully testable

Rule 5 (`chest_aim`) is a WEAPON-READY POSTURE heuristic, not object-level
weapon detection. It reasons purely about where the wrists/elbows sit relative
to the shoulders and hips in 2D image space, so it cannot see a weapon and
cannot tell a rifle from a broomstick. It is deliberately conservative: the
geometry must describe a held, two-handed, forward, chest/chin-height grip and
must persist for several frames before the flag is raised. A real firearm
classifier (custom-trained object model) is the roadmap replacement.
"""
from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# -- COCO 17-keypoint indices --------------------------------------------------
NOSE          = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHLDR, R_SHLDR = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP     = 11, 12
L_KNEE, R_KNEE   = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Skeleton connectivity for drawing
SKELETON_EDGES = [
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
    (L_SHLDR, R_SHLDR),
    (L_SHLDR, L_ELBOW), (R_SHLDR, R_ELBOW),
    (L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST),
    (L_SHLDR, L_HIP),   (R_SHLDR, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),    (R_HIP, R_KNEE),
    (L_KNEE, L_ANKLE),  (R_KNEE, R_ANKLE),
]


@dataclass
class PostureFlags:
    """One instance per tracked person per detection frame."""
    track_id:  int
    keypoints: Optional[np.ndarray] = None  # shape (17, 3): x, y, visibility
    lying:     bool = False
    crouching: bool = False
    vigorous:  bool = False   # head oscillating -- vigorous look-around
    arms_up:   bool = False   # wrists above shoulders (surrender / overhead raise)
    chest_aim: bool = False   # HELD two-handed forward chest/chin-height grip (weapon-ready posture)

    @property
    def any_anomaly(self) -> bool:
        return self.lying or self.crouching or self.vigorous or self.arms_up or self.chest_aim

    @property
    def label(self) -> str:
        tags = []
        if self.lying:      tags.append("LYING")
        if self.crouching:  tags.append("CROUCH")
        if self.vigorous:   tags.append("SCAN")
        if self.arms_up:    tags.append("ARMS-UP")
        if self.chest_aim:  tags.append("AIM")
        return " ".join(tags) if tags else ""

    @property
    def risk_weight(self) -> float:
        """0.0-1.0 anomaly severity fed into risk_engine._behaviour()."""
        if self.lying:      return 1.0
        if self.chest_aim:  return 0.95   # active weapon aim — highest non-lying threat
        if self.arms_up:    return 0.9
        if self.crouching:  return 0.7
        if self.vigorous:   return 0.5
        return 0.0



class PoseEstimator:
    """
    Loads yolo26n-pose.pt once and runs inference on person bounding-box crops.

    Running on crops rather than full frames is dramatically cheaper, and every
    crop from every camera in a detection batch is resized to one fixed square
    size and run in fixed-size batches — a padded batch of 24 crops is ~45 ms
    on an RTX 3050, and stays there whether the scene has 2 people or 15.
    """

    def __init__(self, weights: str = "yolo26n-pose.pt",
                 device: str = "0",
                 half: bool = True,
                 kp_conf: float = 0.5,
                 max_batch: int = 24,
                 imgsz: int = 256) -> None:
        logger.info("Loading pose model: %s", weights)
        self._model   = YOLO(weights)
        self._device  = device
        self._kp_conf = kp_conf
        self._imgsz   = int(imgsz)
        # quantize=16 for FP16; half= is deprecated in ultralytics 8.4.x
        self._q = 16 if half else None

        # Person crops from every camera in a detection batch go through the
        # pose head together, so this batch grows with the crowd (streams x
        # people). Two things keep the cost bounded and — crucially — CONSTANT:
        #   * every crop is resized to a fixed imgsz x imgsz square, so the
        #     input tensor never changes shape with the crop's aspect ratio.
        #     (Feeding raw crops makes cuDNN re-autotune every call: ~3 s vs
        #     ~45 ms — the exact trap the main detector's fixed buckets avoid.)
        #   * each chunk is padded up to a fixed bucket size.
        self._max_batch = max(1, int(max_batch))
        # Two shapes cover it: a small batch for the everyday few-people case and
        # the full cap for a crowd. More buckets just lengthen the warm-up for a
        # few ms of saved pad compute (the pose model is tiny).
        self._buckets = sorted({min(8, self._max_batch), self._max_batch})
        self._pad = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)

        # Warm every bucket shape so no crowd size pays the autotune cost live.
        for b in self._buckets:
            self._model.predict([self._pad] * b, imgsz=self._imgsz,
                                quantize=self._q, device=self._device,
                                verbose=False)
        logger.info("Pose model ready  device=%s  kp_conf=%.2f  imgsz=%d  "
                    "batch buckets %s", device, kp_conf, self._imgsz, self._buckets)

    def run(self, frame: np.ndarray,
            person_bboxes: list[tuple[int, int, int, int, int]]
            ) -> dict[int, np.ndarray]:
        """Single-frame convenience wrapper around :meth:`run_batch`.

        `person_bboxes` is a list of (x1, y1, x2, y2, key); the returned dict is
        keyed by that same `key`. See :meth:`run_batch`.
        """
        return self.run_batch([(frame, person_bboxes)])

    def run_batch(self, per_frame):
        """
        Run pose estimation on person crops taken from several frames in ONE
        batched forward pass.

        Every camera in a detection batch contributes its person crops here, so
        the pose head is called once for the whole batch instead of once per
        camera — the same batching win the main detector already gets.

        Args:
            per_frame: iterable of (frame_bgr, boxes) where boxes is a list of
                       (x1, y1, x2, y2, key). `key` is opaque and only has to be
                       hashable and unique across the batch — the detector uses
                       (cam_id, track_id) so IDs from different cameras don't
                       collide.

        Returns:
            dict mapping each `key` to a keypoints ndarray of shape (17, 3):
            columns are x, y, visibility.  Joints below kp_conf are zeroed.
        """
        S = self._imgsz
        crops: list[np.ndarray] = []
        meta:  list[tuple]      = []   # (key, ox, oy, sx, sy) — sx/sy map S-space back

        for frame, boxes in per_frame:
            if not boxes:
                continue
            h, w = frame.shape[:2]
            for x1, y1, x2, y2, key in boxes:
                cx1 = max(0, int(x1));  cy1 = max(0, int(y1))
                cx2 = min(w, int(x2));  cy2 = min(h, int(y2))
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                crop = frame[cy1:cy2, cx1:cx2]
                ch, cw = crop.shape[:2]
                # Fixed square input — keeps the batch tensor shape constant so
                # cuDNN autotunes once, not every call. Keypoints come back in
                # S-space and are scaled back by (crop_size / S).
                crops.append(cv2.resize(crop, (S, S)))
                meta.append((key, cx1, cy1, cw / S, ch / S))

        if not crops:
            return {}

        # Fixed-shape chunks: pad each up to the next bucket so every predict()
        # call is one of a handful of pre-warmed shapes.
        results = []
        for i in range(0, len(crops), self._max_batch):
            chunk = crops[i:i + self._max_batch]
            n = len(chunk)
            target = next(b for b in self._buckets if b >= n)
            if target > n:
                chunk = chunk + [self._pad] * (target - n)
            out = self._model.predict(chunk, imgsz=S, quantize=self._q,
                                      device=self._device, verbose=False)
            results.extend(out[:n])

        out: dict = {}
        for (key, ox, oy, sx, sy), res in zip(meta, results):
            if res.keypoints is None or res.keypoints.xy is None:
                continue
            xy   = res.keypoints.xy.cpu().numpy()    # (n_dets, 17, 2)
            conf = res.keypoints.conf.cpu().numpy()  # (n_dets, 17)
            if len(xy) == 0:
                continue
            # Pick the detection with the highest mean keypoint confidence
            best     = int(conf.max(axis=1).argmax())
            kp       = np.zeros((17, 3), dtype=np.float32)
            kp[:, 0] = xy[best, :, 0] * sx + ox   # S-space → crop → full frame
            kp[:, 1] = xy[best, :, 1] * sy + oy
            kp[:, 2] = conf[best]
            kp[kp[:, 2] < self._kp_conf, :] = 0.0   # zero low-confidence joints
            out[key] = kp

        return out


class PoseClassifier:
    """
    Stateful, per-track geometry rule engine.

    Every threshold is expressed relative to the person's own body size — the
    torso length (shoulder-mid to hip-mid) and shoulder width — never in raw
    pixels. Raw-pixel rules fired constantly on distant people (a 40 px tall
    walker's knee is always "within 30 px of the hip") and never on close ones.

    Flags are debounced per track: a raw rule must hold for `hold_frames`
    detection frames to switch on, and stays on through `release_frames` missed
    frames, so keypoint jitter neither triggers nor flickers an anomaly.
    All thresholds are tunable via the config.yaml `pose:` block.
    """

    LYING_RATIO_THRESH = 0.45   # visible-skeleton h/w below this (with a horizontal torso) -> lying
    LYING_TORSO_DEG    = 55.0   # torso tilted more than this from vertical -> horizontal body
    CROUCH_KNEE_RATIO  = 0.35   # knee.y - hip.y below this * torso -> thighs folded (crouch/squat)
    CROUCH_HIP_RATIO   = 0.60   # torso shorter than 60% of this track's upright baseline ...
    CROUCH_KNEE_SOFT   = 0.60   # ... AND knee.y - hip.y below this * baseline -> crouching
    SCAN_WINDOW        = 12     # frames of head-yaw history per track
    SCAN_STD_RATIO     = 0.22   # std-dev of (nose_x - shoulder_mid_x) / shoulder_w -> vigorous scan
    SCAN_MIN_FACING    = 0.30   # shoulder_w must be >= this * torso (roughly facing the camera)
    ARMS_UP_MARGIN     = 0.12   # wrist above the shoulder line by this * torso
    ARMS_UP_BOTH       = True   # both hands raised (surrender / overhead); one hand = waving, ignored
    MIN_VISIBLE_KP     = 9      # fewer confident joints than this -> skeleton too unreliable to judge
    MIN_SKELETON_PX    = 70     # skeleton smaller than this in its longest extent (distant person) -> not judged
    HOLD_FRAMES        = 3      # raw rule must persist this many frames to switch a flag on
    RELEASE_FRAMES     = 2      # ... and may drop out this many frames before it switches off

    # -- Rule 5: weapon-ready posture. All ratios are of shoulder width or of
    #    body height (shoulder->hip). Tuned to CATCH a real two-handed gun hold
    #    (high-ready to low-ready, aimed at or across the camera) while still
    #    rejecting hands-at-the-belt, one-hand-up, and folded arms.
    AIM_BAND_ABOVE       = 0.60  # wrist band top: this * body_h ABOVE shoulder (~forehead)
    AIM_BAND_BELOW       = 0.55  # wrist band bottom: this * body_h BELOW shoulder (~navel)
    AIM_WRIST_SEP_RATIO  = 0.85  # wrists within this * shoulder-width of each other
    AIM_WRIST_LEVEL_RATIO = 0.60 # |left wrist.y - right wrist.y| within this * shoulder-width
    AIM_CENTER_MARGIN    = 0.55  # both wrists within this * sh-width of the shoulder span
    AIM_WRIST_ABOVE_HIP  = 0.10  # each wrist at least this * body_h above the hip (arms off the thighs)
    AIM_HOLD_FRAMES      = 3     # raw geometry must persist this many detection frames before firing

    def __init__(self, config: dict | None = None) -> None:
        p = (config or {}).get("pose", {})
        f = lambda k, d: float(p.get(k, d))          # noqa: E731
        i = lambda k, d: int(p.get(k, d))            # noqa: E731
        self.LYING_RATIO_THRESH = f("lying_ratio", self.LYING_RATIO_THRESH)
        self.LYING_TORSO_DEG    = f("lying_torso_deg", self.LYING_TORSO_DEG)
        self.CROUCH_KNEE_RATIO  = f("crouch_knee_ratio", self.CROUCH_KNEE_RATIO)
        self.CROUCH_HIP_RATIO   = f("crouch_ratio", self.CROUCH_HIP_RATIO)
        self.CROUCH_KNEE_SOFT   = f("crouch_knee_soft", self.CROUCH_KNEE_SOFT)
        self.SCAN_WINDOW        = i("scan_window", self.SCAN_WINDOW)
        self.SCAN_STD_RATIO     = f("scan_std_ratio", self.SCAN_STD_RATIO)
        self.SCAN_MIN_FACING    = f("scan_min_facing", self.SCAN_MIN_FACING)
        self.ARMS_UP_MARGIN     = f("arms_up_margin_ratio", self.ARMS_UP_MARGIN)
        self.ARMS_UP_BOTH       = bool(p.get("arms_up_both", self.ARMS_UP_BOTH))
        self.MIN_VISIBLE_KP     = i("min_visible_kp", self.MIN_VISIBLE_KP)
        self.MIN_SKELETON_PX    = f("min_skeleton_px", self.MIN_SKELETON_PX)
        self.HOLD_FRAMES        = max(1, i("hold_frames", self.HOLD_FRAMES))
        self.RELEASE_FRAMES     = max(0, i("release_frames", self.RELEASE_FRAMES))
        self.AIM_BAND_ABOVE       = f("aim_band_above",     self.AIM_BAND_ABOVE)
        self.AIM_BAND_BELOW       = f("aim_band_below",     self.AIM_BAND_BELOW)
        self.AIM_WRIST_SEP_RATIO  = f("aim_wrist_sep",      self.AIM_WRIST_SEP_RATIO)
        self.AIM_WRIST_LEVEL_RATIO = f("aim_wrist_level",   self.AIM_WRIST_LEVEL_RATIO)
        self.AIM_CENTER_MARGIN    = f("aim_center_margin",  self.AIM_CENTER_MARGIN)
        self.AIM_WRIST_ABOVE_HIP  = f("aim_wrist_above_hip", self.AIM_WRIST_ABOVE_HIP)
        self.AIM_HOLD_FRAMES      = max(1, i("aim_hold_frames", self.AIM_HOLD_FRAMES))
        # Per-track state. `key` is whatever the caller passes — the detector
        # uses (cam_id, track_id) so that track id 1 on two different cameras
        # keeps two independent streaks / histories. The tests pass a bare int.
        self._yaw_history:     dict = {}
        self._height_baseline: dict = {}
        self._aim_streak:      dict = {}   # consecutive raw-aim frames per key
        self._debounce:        dict = {}   # key -> {flag: (on_streak, off_streak, active)}

    # -- debounce ------------------------------------------------------------
    def _settle(self, key, name: str, raw: bool) -> bool:
        states = self._debounce.setdefault(key, {})
        on, off, active = states.get(name, (0, 0, False))
        if raw:
            on, off = on + 1, 0
            if on >= self.HOLD_FRAMES:
                active = True
        else:
            on, off = 0, off + 1
            if off > self.RELEASE_FRAMES:
                active = False
        states[name] = (on, off, active)
        return active

    def classify(self, key, kp: np.ndarray) -> PostureFlags:
        """
        kp: (17, 3) array -- x, y, visibility.  visibility == 0 means absent.
        `key` identifies the track (see __init__).
        Returns PostureFlags for this person this frame.
        """
        flags = PostureFlags(track_id=key, keypoints=kp)

        def pt(idx: int) -> Optional[tuple[float, float]]:
            return (float(kp[idx, 0]), float(kp[idx, 1])) if kp[idx, 2] > 0 else None

        def mid(a: int, b: int) -> Optional[tuple[float, float]]:
            pa, pb = pt(a), pt(b)
            if pa and pb:
                return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
            return pa or pb

        shoulders = mid(L_SHLDR, R_SHLDR)
        hips      = mid(L_HIP,   R_HIP)
        knees     = mid(L_KNEE,  R_KNEE)
        nose      = pt(NOSE)
        l_wrist   = pt(L_WRIST);  r_wrist = pt(R_WRIST)
        l_shldr   = pt(L_SHLDR);  r_shldr = pt(R_SHLDR)
        visible   = kp[kp[:, 2] > 0]

        # Distant or half-occluded people give garbage joints (knees snapped to
        # hips, a missing leg); every rule below misfires on them. Their raw
        # rules read False, so an established flag still releases normally.
        reliable = (len(visible) >= self.MIN_VISIBLE_KP and
                    max(float(np.ptp(visible[:, 0])), float(np.ptp(visible[:, 1]))) >= self.MIN_SKELETON_PX)

        torso = None                                   # shoulder-mid -> hip-mid length
        if shoulders and hips:
            torso = max(float(np.hypot(hips[0] - shoulders[0], hips[1] - shoulders[1])), 1.0)
        sh_w = abs(l_shldr[0] - r_shldr[0]) if (l_shldr and r_shldr) else None

        # -- Rule 1: Lying on the ground ----------------------------------------
        # The torso itself must be horizontal. A wide keypoint cloud alone is
        # not enough: arms spread out or a partial (cropped) skeleton are wide too.
        raw_lying = False
        if reliable and shoulders and hips:
            dx = abs(hips[0] - shoulders[0])
            dy = abs(hips[1] - shoulders[1])
            tilt = float(np.degrees(np.arctan2(dx, max(dy, 1e-3))))   # 0 = upright
            kp_h = float(visible[:, 1].max() - visible[:, 1].min())
            kp_w = float(visible[:, 0].max() - visible[:, 0].min())
            flat = kp_w > 1 and (kp_h / kp_w) < self.LYING_RATIO_THRESH
            raw_lying = tilt > self.LYING_TORSO_DEG and flat
        flags.lying = self._settle(key, "lying", raw_lying)

        # -- Rule 2: Crouching ---------------------------------------------------
        # (a) thighs folded: knees risen to near hip height, relative to torso;
        # (b) body compressed versus this track's own upright baseline, with the
        #     knees at least partly raised (a torso merely foreshortened by
        #     walking away from the camera does not count).
        raw_crouch = False
        ankles = [a for a in (pt(L_ANKLE), pt(R_ANKLE)) if a is not None]
        both_legs = all(pt(j) is not None for j in (L_HIP, R_HIP, L_KNEE, R_KNEE))
        if reliable and not flags.lying and torso and knees and shoulders and hips:
            knee_drop = knees[1] - hips[1]              # +ve = knees below hips
            upright = knee_drop > 0.7 * torso and abs(hips[0] - shoulders[0]) < 0.5 * torso
            if upright:
                prev = self._height_baseline.get(key)
                self._height_baseline[key] = torso if prev is None else 0.8 * prev + 0.2 * torso
            baseline = self._height_baseline.get(key)
            # Folded thighs need the whole lower body in view: both hips and
            # knees, and a foot planted below the knees. A top-down view of
            # someone walking downhill foreshortens the thigh the same way.
            folded = (both_legs and bool(ankles)
                      and all(a[1] > knees[1] + 0.25 * torso for a in ankles)
                      and knee_drop < self.CROUCH_KNEE_RATIO * torso)
            compressed = (baseline is not None and torso < baseline * self.CROUCH_HIP_RATIO
                          and knee_drop < self.CROUCH_KNEE_SOFT * baseline)
            raw_crouch = folded or compressed
        flags.crouching = self._settle(key, "crouching", raw_crouch)

        # -- Rule 3: Vigorous head scan -----------------------------------------
        # Head yaw = nose offset from the shoulder centre, in shoulder widths.
        # Walking moves nose AND shoulders together, so it no longer counts;
        # only the head swinging left/right against the body does.
        raw_scan = False
        if reliable and nose and shoulders and sh_w and torso and sh_w >= self.SCAN_MIN_FACING * torso:
            hist = self._yaw_history.setdefault(
                key, collections.deque(maxlen=self.SCAN_WINDOW))
            hist.append((nose[0] - shoulders[0]) / sh_w)
            if len(hist) >= max(4, self.SCAN_WINDOW // 2):
                raw_scan = float(np.std(hist)) > self.SCAN_STD_RATIO
        flags.vigorous = self._settle(key, "vigorous", raw_scan)

        # -- Rule 4: Arms raised / surrender posture -----------------------------
        raw_arms = False
        if reliable and (l_shldr or r_shldr):
            scale = torso or (sh_w * 1.6 if sh_w else 40.0)
            margin = self.ARMS_UP_MARGIN * scale
            up_l = bool(l_wrist and l_shldr and l_wrist[1] < l_shldr[1] - margin)
            up_r = bool(r_wrist and r_shldr and r_wrist[1] < r_shldr[1] - margin)
            raw_arms = (up_l and up_r) if self.ARMS_UP_BOTH else (up_l or up_r)
        flags.arms_up = self._settle(key, "arms_up", raw_arms)

        # -- Rule 5: Weapon-ready posture (2D-skeleton heuristic) -------------
        # A two-handed gun hold: both hands on one object, held out from the
        # body somewhere between forehead and navel. Clauses:
        #   in_band       wrists between ~forehead and ~navel height
        #   wrists_level   hands roughly the same height (two on one object)
        #   wrists_close   hands within ~0.85 shoulder-width of each other
        #   wrists_centred hands in front of the torso (rejects folded arms,
        #                  where a wrist sits out past the far shoulder)
        #   off_thighs     arms raised off the legs
        #   arms_engaged   at least one elbow up near torso height, i.e. the
        #                  arm is bent/forward, not hanging at the side
        raw_aim = False
        l_elb = pt(L_ELBOW);  r_elb = pt(R_ELBOW)
        if l_wrist and r_wrist and l_shldr and r_shldr and not flags.lying:
            sh_y     = (l_shldr[1] + r_shldr[1]) / 2
            sh_xmin  = min(l_shldr[0], r_shldr[0])
            sh_xmax  = max(l_shldr[0], r_shldr[0])
            sh_width = max(sh_xmax - sh_xmin, 20)
            hip_y    = hips[1] if hips else sh_y + sh_width * 2
            body_h   = max(abs(hip_y - sh_y), 30)

            band_top   = sh_y - body_h * self.AIM_BAND_ABOVE
            band_bot   = sh_y + body_h * self.AIM_BAND_BELOW
            wrist_avg_y = (l_wrist[1] + r_wrist[1]) / 2
            m = sh_width * self.AIM_CENTER_MARGIN

            in_band       = band_top <= wrist_avg_y <= band_bot
            wrists_level  = abs(l_wrist[1] - r_wrist[1]) < sh_width * self.AIM_WRIST_LEVEL_RATIO
            wrists_close  = abs(l_wrist[0] - r_wrist[0]) < sh_width * self.AIM_WRIST_SEP_RATIO
            wrists_centred = (sh_xmin - m <= l_wrist[0] <= sh_xmax + m and
                              sh_xmin - m <= r_wrist[0] <= sh_xmax + m)
            off_thighs    = (l_wrist[1] < hip_y - body_h * self.AIM_WRIST_ABOVE_HIP and
                             r_wrist[1] < hip_y - body_h * self.AIM_WRIST_ABOVE_HIP)

            elbows = [e for e in (l_elb, r_elb) if e is not None]
            if elbows:
                # engaged = elbow no lower than mid-torso (bent/forward arm).
                arms_engaged = any(e[1] <= sh_y + body_h * 0.55 for e in elbows)
            else:
                arms_engaged = True                            # no elbow data — don't block on it

            raw_aim = (in_band and wrists_level and wrists_close
                       and wrists_centred and off_thighs and arms_engaged)

        streak = self._aim_streak.get(key, 0) + 1 if raw_aim else 0
        self._aim_streak[key] = streak
        flags.chest_aim = streak >= self.AIM_HOLD_FRAMES

        return flags


    def flush_track(self, key) -> None:
        """Remove stale per-track state when a ByteTrack ID is lost."""
        self._yaw_history.pop(key, None)
        self._height_baseline.pop(key, None)
        self._aim_streak.pop(key, None)
        self._debounce.pop(key, None)

    def known_keys(self) -> set:
        """Every track key the classifier currently holds state for."""
        return (set(self._aim_streak)
                | set(self._yaw_history)
                | set(self._height_baseline)
                | set(self._debounce))

    def flush_missing(self, live_keys) -> None:
        """Drop per-track state for any key not in `live_keys`.

        Called once per detection batch with the keys seen this pass, so the
        per-track dicts can't grow without bound as ByteTrack ids churn and a
        stale aim streak can't carry over to a reused id.
        """
        live = set(live_keys)
        for key in self.known_keys() - live:
            self.flush_track(key)


# -- Skeleton overlay drawing --------------------------------------------------
def draw_skeleton(canvas: np.ndarray, flags: PostureFlags,
                  dim: bool = False) -> np.ndarray:
    """Draw 17-point skeleton + anomaly label onto canvas in-place."""
    kp = flags.keypoints
    if kp is None:
        return canvas

    # Red for any anomaly; cyan for normal; dimmed on carried (non-infer) frames
    colour: tuple[int, int, int] = (0, 0, 220) if flags.any_anomaly else (220, 200, 0)
    if dim:
        colour = tuple(int(c * 0.6) for c in colour)

    # Joints
    for i in range(17):
        if kp[i, 2] > 0:
            cv2.circle(canvas, (int(kp[i, 0]), int(kp[i, 1])), 4, colour, -1)

    # Limb edges
    for a, b in SKELETON_EDGES:
        if kp[a, 2] > 0 and kp[b, 2] > 0:
            cv2.line(canvas,
                     (int(kp[a, 0]), int(kp[a, 1])),
                     (int(kp[b, 0]), int(kp[b, 1])),
                     colour, 2, cv2.LINE_AA)

    # Anomaly text label above the person
    label = flags.label
    if label:
        visible = kp[kp[:, 2] > 0]
        if len(visible):
            tx = int(visible[:, 0].min())
            ty = max(int(visible[:, 1].min()) - 8, 14)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas,
                          (tx - 2, ty - th - 4), (tx + tw + 4, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(canvas, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 1, cv2.LINE_AA)

    return canvas
