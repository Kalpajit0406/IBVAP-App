"""
Batched multi-stream detector.

Three techniques keep GPU cost sub-linear in the number of cameras:

  1. Batched inference — frames from every camera due for a detection pass are
     stacked into one forward pass, the way DeepStream's nvstreammux does. One
     batched call on N frames costs far less than N single-frame calls, because
     a single 640x640 image leaves most of the GPU's SMs idle.

  2. Motion gating — a camera whose scene has not changed never enters the
     batch at all (see motion_gate.py).

  3. Adaptive skipping with full-rate tracking — detection runs at a target
     rate below the stream frame rate, while tracks are carried forward on
     every frame by linear extrapolation, so boxes still move smoothly on the
     monitor between detector passes.
"""
from __future__ import annotations

import copy
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from .motion_gate import MotionGate
from .posture import PoseEstimator, PoseClassifier, PostureFlags, draw_skeleton

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*ByteTrack.*", category=FutureWarning)

_PERSON = 0
_VEHICLES = {2, 3, 5, 7}          # car, motorcycle, bus, truck


@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    confidence: float
    is_person: bool
    is_vehicle: bool
    velocity: tuple[float, float] = (0.0, 0.0)   # px/frame, for extrapolation
    posture: Optional[PostureFlags] = None        # set by pose pass; None on non-persons
    plate: Optional[str] = None                   # set by ANPR on vehicles
    plate_conf: float = 0.0
    vehicle_type: Optional[str] = None            # set by VehicleTypeClassifier
    vehicle_type_conf: float = 0.0
    face_name: Optional[str] = None               # matched watchlist identity, once its burst finalizes
    face_conf: float = 0.0                        # similarity of that match
    on_watchlist: bool = False                    # True only while the match is still TTL-fresh


@dataclass
class StreamResult:
    cam_id: int
    timestamp: float
    frame: np.ndarray
    detections: list[Detection]
    person_count: int
    vehicle_count: int
    annotated_frame: Optional[np.ndarray] = None
    inferred: bool = False        # True when the detector actually ran
    gated: bool = False           # True when motion gating skipped it
    weapons: list = field(default_factory=list)   # WeaponHit list (src/weapon.py); [] on carried frames
    faces: list = field(default_factory=list)     # FaceHit list (ibvap/face.py); [] on carried frames


@dataclass
class _CamState:
    """State kept across frames for one camera stream."""
    tracker: sv.ByteTrack
    gate: MotionGate
    frames: int = 0
    last_detect_tick: int = -999
    last_gate_pass_tick: int = -999
    detections: list[Detection] = field(default_factory=list)
    prev_centres: dict[int, tuple[float, float]] = field(default_factory=dict)
    person_count: int = 0
    vehicle_count: int = 0
    infer_count: int = 0
    gated_count: int = 0



@dataclass
class PipelineStats:
    frames_in: int = 0
    detector_passes: int = 0      # batched calls issued
    frames_inferred: int = 0      # frames that went through the model
    frames_gated: int = 0         # skipped by motion gate
    frames_rate_skipped: int = 0  # skipped by the detection-rate limiter
    batch_total: int = 0          # sum of batch sizes, for the mean
    last_batch: int = 0
    padded_frames: int = 0        # blanks added to reach a fixed batch shape
    infer_ms: float = 0.0

    @property
    def mean_batch(self) -> float:
        if not self.detector_passes:
            return 0.0
        return self.batch_total / self.detector_passes

    @property
    def gpu_saving_pct(self) -> float:
        """Share of incoming frames the GPU never had to look at."""
        if not self.frames_in:
            return 0.0
        return 100.0 * (self.frames_gated + self.frames_rate_skipped) / self.frames_in


def _wrists(d) -> list:
    """Visible wrist points (x, y) from a person's pose, for the weapon
    engine's "is the gun in a hand" check. Empty when pose is unknown."""
    kp = getattr(getattr(d, "posture", None), "keypoints", None)
    if kp is None:
        return []
    return [(float(kp[j, 0]), float(kp[j, 1])) for j in (9, 10) if kp[j, 2] > 0]


class Detector:
    """One YOLO model shared by every camera; per-camera tracker and motion gate."""

    @classmethod
    def from_profile(cls, config: dict, profile: str, num_cameras: int = 0,
                     anpr=None, calibrator=None, weapon=None, vehicle_type=None,
                     anpr_results=None, vehicle_arrival=None,
                     native_frame_fn=None, face=None, face_arrival=None,
                     face_results=None) -> "Detector":
        """Build a Detector using a named model_profiles entry.

        `anpr` / `calibrator` / `weapon` / `vehicle_type` / `anpr_results` /
        `vehicle_arrival` / `face` / `face_arrival` / `face_results` — pass
        existing instances to reuse them across a hot-swap instead of losing
        their per-camera state.
        """
        profiles = config.get("model_profiles", {})
        if profile not in profiles:
            raise ValueError(f"Unknown profile {profile!r}. Available: {list(profiles)}")
        p = profiles[profile]
        patched = copy.deepcopy(config)
        patched.setdefault("model", {})
        patched["model"]["weights"] = p["det_weights"]
        patched["model"]["max_batch"] = p.get("max_batch", patched["model"].get("max_batch", 8))
        patched.setdefault("pose", {})
        patched["pose"]["weights"] = p["pose_weights"]
        return cls(patched, num_cameras=num_cameras, anpr=anpr,
                   calibrator=calibrator, weapon=weapon, vehicle_type=vehicle_type,
                   anpr_results=anpr_results, vehicle_arrival=vehicle_arrival,
                   native_frame_fn=native_frame_fn, face=face,
                   face_arrival=face_arrival, face_results=face_results)

    @classmethod
    def from_weights(cls, config: dict, det_weights: str, pose_weights: str,
                     label: str = "", num_cameras: int = 0, anpr=None,
                     calibrator=None, weapon=None, vehicle_type=None,
                     anpr_results=None, vehicle_arrival=None,
                     native_frame_fn=None, face=None, face_arrival=None,
                     face_results=None) -> "Detector":
        """Build a Detector from explicit weight paths (a fine-tuned model from
        the continuous-learning registry). Paths are allowlisted to `models/`
        or the repo-root `yolo26*.pt` files."""
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        allowed = {(root / "models").resolve()}
        for w in (det_weights, pose_weights):
            rp = Path(w).resolve()
            ok = rp.parent in allowed or (rp.parent == root and rp.name.startswith("yolo26"))
            if not ok:
                raise ValueError(f"weights path not allowed: {w}")
        patched = copy.deepcopy(config)
        patched.setdefault("model", {})["weights"] = det_weights
        patched["model"]["max_batch"] = patched["model"].get("max_batch", 8)
        patched.setdefault("pose", {})["weights"] = pose_weights
        return cls(patched, num_cameras=num_cameras, anpr=anpr,
                   calibrator=calibrator, weapon=weapon, vehicle_type=vehicle_type,
                   anpr_results=anpr_results, vehicle_arrival=vehicle_arrival,
                   native_frame_fn=native_frame_fn, face=face,
                   face_arrival=face_arrival, face_results=face_results)

    def __init__(self, config: dict, num_cameras: int = 0, anpr=None,
                 calibrator=None, weapon=None, vehicle_type=None,
                 anpr_results=None, vehicle_arrival=None,
                 native_frame_fn=None, face=None, face_arrival=None,
                 face_results=None) -> None:
        m = config["model"]
        self._weights = str(m["weights"])
        # A TensorRT .engine (or .onnx) is already bound to its device and only
        # supports predict/val — .to(), .train() etc. raise. Track this so the
        # PyTorch-only calls below are skipped.
        self._is_engine = self._weights.lower().endswith((".engine", ".onnx", ".plan"))
        self._model = YOLO(self._weights)
        self._conf = m.get("confidence", 0.40)
        self._iou = m.get("iou", 0.50)
        self._imgsz = m.get("image_size", 640)
        self._half = bool(m.get("half", False))
        self._max_batch = int(m.get("max_batch", 16))
        # Upper bound on boxes returned per frame. The COCO/Ultralytics default
        # is 300 — kept explicit so a crowded scene (10-15+ people plus
        # vehicles) is never silently truncated, and so it is tunable.
        self._max_det = int(m.get("max_det", 300))

        # ── Detection scope + per-class confidence ──────────────────────────
        # Restrict inference to the classes we care about (person + 4 vehicle
        # types) — cleaner output, a touch less NMS. Then a per-class floor:
        # a lower bar for people (they are what matters most) and a higher bar
        # for vehicles (cuts parked-car / reflection false positives).
        self._classes = list(m.get("classes", [_PERSON, *sorted(_VEHICLES)]))
        self._person_conf = float(m.get("person_conf", 0.30))
        self._vehicle_conf = float(m.get("vehicle_conf", 0.40))
        # predict() floor = the lowest per-class bar, refined afterwards.
        self._predict_conf = min(self._conf, self._person_conf, self._vehicle_conf)

        # ── Adaptive calibration (per-camera, live, no training) ────────────
        cal_cfg = (config.get("learning", {}) or {}).get("calibration", {}) or {}
        if calibrator is not None:
            self._calibrator = calibrator
        elif cal_cfg.get("enabled"):
            from .calibration import Calibrator
            self._calibrator = Calibrator(config)
        else:
            self._calibrator = None
        if self._calibrator is not None:
            # give calibration headroom to LOOSEN a floor, not just raise it
            self._predict_conf = max(0.05, self._predict_conf - float(cal_cfg.get("headroom", 0.05)))

        tk = config.get("tracker", {}) or {}
        self._tracker_kwargs = {
            "track_activation_threshold": float(tk.get("activation_conf", 0.25)),
            "lost_track_buffer": int(tk.get("lost_buffer_frames", 60)),
            "minimum_matching_threshold": float(tk.get("match_thresh", 0.85)),
            "frame_rate": max(1, int(float(m.get("detect_fps", 8.0)))),
        }

        # Detection rate: how many detector passes per second per camera.
        # Streams run at 24-25 fps; detecting at 8 is plenty when tracking
        # carries the boxes between passes.
        self._detect_fps = float(m.get("detect_fps", 8.0))
        self._stream_fps = float(m.get("stream_fps", 24.0))
        self._detect_every = max(1, round(self._stream_fps / max(self._detect_fps, 0.1)))

        self._motion_gating = bool(m.get("motion_gating", True))
        gate_cfg = m.get("motion", {}) or {}
        # force_every / hold_frames count gate calls, not raw frames: the gate
        # only sees frames that already cleared the rate cap.
        self._gate_kwargs = {
            "pixel_threshold": gate_cfg.get("pixel_threshold", 18),
            "area_threshold": gate_cfg.get("area_threshold", 0.002),
            "force_every": gate_cfg.get("force_every", max(1, int(self._detect_fps))),
            "hold_frames": gate_cfg.get("hold_frames", 3),
        }

        # Fixed batch sizes we are willing to submit; anything between is padded
        # up to the next one. Set before the device warm-up, which primes
        # exactly these shapes. See _bucket for why this matters so much.
        #
        # A TensorRT engine is built with ONE fixed input shape
        # (max_batch, 3, imgsz, imgsz), so there is exactly one legal bucket:
        # always pad to max_batch. A PyTorch model tolerates any shape, so it
        # gets the finer ladder to avoid wasting compute on small batches.
        if self._is_engine:
            self._buckets = [self._max_batch]
        else:
            self._buckets = [b for b in (1, 2, 4, 8, 16, 32) if b <= self._max_batch]
            if not self._buckets:
                self._buckets = [self._max_batch]
            elif self._buckets[-1] < self._max_batch:
                self._buckets.append(self._max_batch)
        self._pad_frame: Optional[np.ndarray] = None

        self._device = self._resolve_device(str(m.get("device", "0")))

        self._cams: dict[int, _CamState] = {}
        self._box_ann = sv.BoxAnnotator(thickness=2)
        self._lbl_ann = sv.LabelAnnotator(text_scale=0.45, text_thickness=1)
        self.stats = PipelineStats()
        # One tick per batch, shared by every camera. Scheduling detection on a
        # global clock rather than each camera's own frame count is what makes
        # cameras come due together — and a batch only pays if its members
        # arrive in the same pass.
        self._tick = 0

        # ── Pose estimation (off when pose.enabled is false) ──────────────────
        pose_cfg = config.get("pose", {})
        self._pose_enabled = bool(pose_cfg.get("enabled", True))
        # Cap pose crops run per camera per pass. 10-15 people are covered with
        # headroom; a pathological crowd degrades gracefully (nearest/biggest
        # first) instead of stalling the inference thread.
        self._pose_max_persons = int(pose_cfg.get("max_persons", 24))
        if self._pose_enabled:
            self._pose = PoseEstimator(
                weights=pose_cfg.get("weights", "yolo26n-pose.pt"),
                device=self._device,
                half=self._half,
                kp_conf=float(pose_cfg.get("kp_conf", 0.5)),
                max_batch=int(pose_cfg.get("max_batch", 24)),
                imgsz=int(pose_cfg.get("imgsz", 256)),
            )
            self._classifier = PoseClassifier(config)
        else:
            self._pose       = None
            self._classifier = None
            logger.info("Pose estimation disabled (pose.enabled: false)")

        # ── ANPR (off unless anpr.enabled and easyocr installed) ─────────────
        anpr_cfg = config.get("anpr", {}) or {}
        if anpr is not None:
            self.anpr = anpr                       # reused across a hot-swap
        elif anpr_cfg.get("enabled"):
            from .anpr import AnprEngine
            eng = AnprEngine(anpr_cfg, device=self._device)
            self.anpr = eng if eng.available else None
        else:
            self.anpr = None

        # ── Weapon detector (off unless weapon.enabled and weights present) ──
        weapon_cfg = config.get("weapon", {}) or {}
        if weapon is not None:
            self.weapon = weapon                   # reused across a hot-swap
        elif weapon_cfg.get("enabled"):
            from .weapon import WeaponDetector
            weng = WeaponDetector(weapon_cfg, device=self._device,
                                  max_batch=self._max_batch)
            self.weapon = weng if weng.available else None
        else:
            self.weapon = None

        # ── Vehicle-type classifier (coarse COCO fallback until a trained
        #    model exists — see training/train_vehicle_type.py) ──────────────
        vt_cfg = config.get("vehicle_type", {}) or {}
        if vehicle_type is not None:
            self.vehicle_type = vehicle_type       # reused across a hot-swap
        elif vt_cfg.get("enabled", True):
            from .vehicle_type import VehicleTypeClassifier
            self.vehicle_type = VehicleTypeClassifier(vt_cfg, device=self._device)
        else:
            self.vehicle_type = None

        # ── Event-triggered ANPR: arrival tracker + dedicated output folder ──
        from .anpr_events import VehicleArrivalTracker, AnprResultWriter
        self._vehicle_arrival = (vehicle_arrival if vehicle_arrival is not None
                                 else VehicleArrivalTracker(
                                     follow_up_max=int(anpr_cfg.get("arrival_followup_reads", 8)),
                                     follow_up_s=float(anpr_cfg.get("arrival_followup_s", 6.0))))
        self.anpr_results = (anpr_results if anpr_results is not None
                             else AnprResultWriter(config))
        # cam_id -> () -> (ts, native_frame) | None. Set by server.py from the
        # live capture objects; only ever called once per vehicle arrival.
        self._native_frame_fn = native_frame_fn
        self._vehicle_passes = 0
        self._vehicle_types: dict[tuple[int, int], "VehicleTypeResult"] = {}

        # ── Face recognition (off unless face.enabled and gallery+weights
        #    present) — a small curated watchlist gallery, not a trained model
        #    fine-tune; see ibvap/face.py, docs/FACE_RECOGNITION.md ───────────
        face_cfg = config.get("face", {}) or {}
        if face is not None:
            self.face = face                       # reused across a hot-swap
        elif face_cfg.get("enabled"):
            from .face import FaceEngine
            feng = FaceEngine(face_cfg, device=self._device)
            self.face = feng if feng.available else None
        else:
            self.face = None

        # ── Event-triggered face capture: per-person burst tracker + writer ──
        from .face_events import PersonArrivalTracker, FaceResultWriter
        self._face_arrival = (face_arrival if face_arrival is not None
                              else PersonArrivalTracker(
                                  burst_n=int(face_cfg.get("burst_n", 5)),
                                  max_attempts=int(face_cfg.get("max_attempts", 15)),
                                  attempt_window_s=float(face_cfg.get("attempt_window_s", 8.0)),
                                  vote_min=int(face_cfg.get("vote_min", 3)),
                                  hit_ttl_s=float(face_cfg.get("hit_ttl_s", 4.0))))
        self.face_results = (face_results if face_results is not None
                             else FaceResultWriter(config))
        self._face_use_native = bool(face_cfg.get("use_native_frame", True))
        self._face_detect_every = max(1, int(face_cfg.get("detect_every", 2)))
        self._face_passes = 0
        self._face_arrivals: list[dict] = []   # finalized burst records, drained by server.py

        logger.info(
            "Detector ready: %s | device=%s | imgsz=%d | conf=%.2f | fp16=%s | "
            "detect every %d frames (%.0f fps of %.0f) | motion gating=%s | "
            "batch buckets %s",
            m["weights"], self._device, self._imgsz, self._conf, self._half,
            self._detect_every, self._detect_fps, self._stream_fps,
            self._motion_gating, self._buckets,
        )

    # ── Setup ────────────────────────────────────────────────────────────────
    def _resolve_device(self, requested: str) -> str:
        if requested != "cpu" and not torch.cuda.is_available():
            logger.warning(
                "CUDA unavailable (torch=%s) — falling back to CPU. Install the CUDA "
                "build: pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu126", torch.__version__)
            return "cpu"

        if requested != "cpu":
            logger.info("CUDA active: %s | torch=%s | VRAM=%.1f GB | backend=%s",
                        torch.cuda.get_device_name(0), torch.__version__,
                        torch.cuda.get_device_properties(0).total_memory / 1e9,
                        "TensorRT engine" if self._is_engine else "PyTorch")
            if not self._is_engine:
                # PyTorch model: move it to the GPU. An engine is already there
                # and .to() would raise.
                self._model.to(f"cuda:{requested}" if requested.isdigit() else requested)
            self._warm_up(requested)
        return requested

    def _warm_up(self, device: str) -> None:
        """
        Prime every batch shape the pipeline will actually submit.

        cuDNN autotunes kernels per input shape on first use, and that tuning
        costs far more than a steady-state pass. Motion gating makes the batch
        size vary from frame to frame, so warming only one shape leaves the
        rest to be tuned mid-flight — which shows up as a slow, jittery first
        few seconds on the monitor.
        """
        t0 = time.perf_counter()
        blank = np.zeros((self._imgsz, self._imgsz, 3), np.uint8)
        kw = dict(device=device, imgsz=self._imgsz, verbose=False)
        if not self._is_engine:
            kw["quantize"] = 16 if self._half else None
        # Only the bucket sizes are ever submitted; prime each a few times so
        # TRT/cuDNN lazy kernel init is done before the first real frame.
        for n in self._buckets:
            for _ in range(5 if self._is_engine else 1):
                self._model.predict([blank] * n, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("GPU warm-up complete in %.0f ms (batch shapes %s)",
                    (time.perf_counter() - t0) * 1000,
                    ", ".join(str(b) for b in self._buckets))

    @property
    def device(self) -> str:
        return self._device

    def _new_tracker(self) -> "sv.ByteTrack":
        try:
            return sv.ByteTrack(**self._tracker_kwargs)
        except TypeError:
            # Older supervision: different kwarg names — fall back to defaults.
            return sv.ByteTrack()

    def _cam(self, cam_id: int) -> _CamState:
        st = self._cams.get(cam_id)
        if st is None:
            st = _CamState(tracker=self._new_tracker(),
                           gate=MotionGate(**self._gate_kwargs))
            self._cams[cam_id] = st
            logger.info("CAM-%02d registered with the pipeline", cam_id)
        return st

    # ── Main entry point ─────────────────────────────────────────────────────
    def process_batch(
        self, items: list[tuple[int, float, np.ndarray]]
    ) -> list[StreamResult]:
        """
        Take one frame from each active camera and return a result for each.

        Frames that clear both the rate limiter and the motion gate are stacked
        into a single batched forward pass; the rest are carried forward by
        track extrapolation, which costs no GPU at all.
        """
        if not items:
            return []

        self.stats.frames_in += len(items)
        self._tick += 1

        to_infer: list[tuple[int, float, np.ndarray]] = []
        to_carry: list[tuple[int, float, np.ndarray]] = []

        for cam_id, ts, frame in items:
            st = self._cam(cam_id)
            st.frames += 1

            # 1. Detection-rate limiter, on the shared tick so every camera
            #    becomes due in the same pass and the batch stays full.
            if self._tick - st.last_detect_tick < self._detect_every:
                self.stats.frames_rate_skipped += 1
                to_carry.append((cam_id, ts, frame))
                continue

            # 2. Motion gate — sub-millisecond, still far cheaper than the GPU.
            if self._motion_gating and not st.gate.should_detect(frame):
                self.stats.frames_gated += 1
                st.gated_count += 1
                # Advance the schedule as though a pass had happened, so a
                # gated camera waits a full interval instead of retrying on the
                # next frame and drifting out of phase with the others.
                st.last_detect_tick = self._tick
                to_carry.append((cam_id, ts, frame))
                continue

            # Only real motion (not the keep-alive sweep) marks a camera "live".
            if not self._motion_gating or st.gate.last_pass_was_motion:
                st.last_gate_pass_tick = self._tick
            to_infer.append((cam_id, ts, frame))

        results: list[StreamResult] = []

        # 3. One batched forward pass for every camera that needs one.
        for start in range(0, len(to_infer), self._max_batch):
            chunk = to_infer[start:start + self._max_batch]
            results.extend(self._infer_batch(chunk))

        for cam_id, ts, frame in to_carry:
            results.append(self._carry_forward(cam_id, ts, frame))

        results.sort(key=lambda r: r.cam_id)
        return results

    # ── Batched inference ────────────────────────────────────────────────────
    def _bucket(self, n: int) -> int:
        """Round a batch up to the next size we are willing to submit.

        Changing the batch shape is expensive: Ultralytics reconfigures its
        predictor and cuDNN re-tunes kernels, and measured on an RTX 3050 a
        run of varying batches averaged 304 ms per pass where a constant
        batch of 12 took 46 ms. Motion gating makes the natural batch size
        jitter with however many cameras happen to be moving, so we snap it
        to a handful of fixed sizes and pad the difference with a blank
        frame. The padding is wasted GPU work, but far cheaper than the
        reconfiguration it avoids.
        """
        for b in self._buckets:
            if n <= b:
                return b
        return self._buckets[-1]

    def _infer_batch(self, chunk: list[tuple[int, float, np.ndarray]]) -> list[StreamResult]:
        frames = [f for _, _, f in chunk]
        real = len(frames)

        target = self._bucket(real)
        if target > real:
            if self._pad_frame is None or self._pad_frame.shape != frames[0].shape:
                self._pad_frame = np.zeros_like(frames[0])
            frames = frames + [self._pad_frame] * (target - real)

        # An engine's precision is baked in at build time; passing quantize= to
        # it makes Ultralytics do extra per-call work and adds latency jitter
        # (measured: p90 39 ms -> 31 ms, max 66 ms -> 33 ms at batch 8).
        predict_kw = dict(conf=self._predict_conf, iou=self._iou,
                          device=self._device, imgsz=self._imgsz,
                          classes=self._classes, max_det=self._max_det,
                          verbose=False)
        if not self._is_engine:
            predict_kw["quantize"] = 16 if self._half else None

        t0 = time.perf_counter()
        outputs = self._model.predict(frames, **predict_kw)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.stats.detector_passes += 1
        self.stats.frames_inferred += real
        self.stats.padded_frames += target - real
        self.stats.batch_total += real
        self.stats.last_batch = real
        self.stats.infer_ms = elapsed_ms / max(real, 1)

        # 1. Track + build a Detection list for every camera in the batch.
        staged: list[tuple[int, float, np.ndarray, list[Detection]]] = [
            (cam_id, ts, frame, self._track(cam_id, out))
            for (cam_id, ts, frame), out in zip(chunk, outputs[:real])
        ]

        # 2. Pose and ANPR each as ONE batched forward pass across every
        #    camera — not one call per camera. The main detector was already
        #    batched this way; these two were not, which made them the real
        #    cost as the stream count grew.
        self._pose_pass(staged)
        self._vehicle_pass(staged)
        weapons_by_cam = self._weapon_pass(staged)
        faces_by_cam = self._face_pass(staged)

        # 3. Draw overlays and package one StreamResult per camera.
        return [
            StreamResult(
                cam_id, ts, frame, dets,
                sum(1 for d in dets if d.is_person),
                sum(1 for d in dets if d.is_vehicle),
                self._annotate(frame, dets, weapons=weapons_by_cam.get(cam_id),
                               faces=faces_by_cam.get(cam_id)),
                inferred=True, weapons=weapons_by_cam.get(cam_id, []),
                faces=faces_by_cam.get(cam_id, []),
            )
            for cam_id, ts, frame, dets in staged
        ]

    def _track(self, cam_id: int, out) -> list[Detection]:
        """YOLO output → per-camera ByteTrack update → Detection list.

        Just the per-camera bookkeeping — no pose, ANPR or drawing, so the
        batch-wide passes can run once over every camera afterwards.
        """
        st = self._cam(cam_id)
        st.last_detect_tick = self._tick
        st.infer_count += 1

        sv_det = sv.Detections.from_ultralytics(out)
        oh, ow = getattr(out, "orig_shape", (720, 1280))

        cal = self._calibrator
        p_bar = cal.person_floor(cam_id, self._person_conf) if cal else self._person_conf
        v_bar = cal.vehicle_floor(cam_id, self._vehicle_conf) if cal else self._vehicle_conf

        # Per-class confidence floor (calibrated). Filter BEFORE the tracker so
        # it never opens a track on a low-confidence, off-target, or
        # learned-noisy-region box.
        if len(sv_det):
            def _keep(i) -> bool:
                c = int(sv_det.class_id[i]); cf = float(sv_det.confidence[i])
                bar = p_bar if c == _PERSON else v_bar if c in _VEHICLES else 1.0
                if cf < bar:
                    return False
                if cal is not None and cf < bar + 0.10:
                    bb = sv_det.xyxy[i]
                    if cal.is_suppressed(cam_id, cal.gp_norm(bb, ow, oh)):
                        return False
                    if not cal.area_ok(cam_id, c, cal.area_norm(bb, ow, oh)):
                        return False
                return True
            keep = np.fromiter((_keep(i) for i in range(len(sv_det))),
                               dtype=bool, count=len(sv_det))
            sv_det = sv_det[keep]

        sv_det = st.tracker.update_with_detections(sv_det)

        detections: list[Detection] = []
        centres: dict[int, tuple[float, float]] = {}

        for i in range(len(sv_det)):
            cid = int(sv_det.class_id[i])
            tid = int(sv_det.tracker_id[i]) if sv_det.tracker_id is not None else -1
            x1, y1, x2, y2 = sv_det.xyxy[i].astype(int)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Velocity from the previous detection pass, so carried frames can
            # move the box instead of freezing it.
            vx = vy = 0.0
            prev = st.prev_centres.get(tid)
            if prev is not None and self._detect_every > 0:
                vx = (cx - prev[0]) / self._detect_every
                vy = (cy - prev[1]) / self._detect_every
            centres[tid] = (cx, cy)

            detections.append(Detection(
                track_id=tid, class_id=cid, class_name=out.names[cid],
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(sv_det.confidence[i]),
                is_person=(cid == _PERSON), is_vehicle=(cid in _VEHICLES),
                velocity=(vx, vy),
            ))

        st.prev_centres = centres
        st.detections = detections
        st.person_count = sum(1 for d in detections if d.is_person)
        st.vehicle_count = sum(1 for d in detections if d.is_vehicle)
        if cal is not None:
            cal.observe(cam_id, detections,
                        {d.track_id for d in detections if d.track_id >= 0}, ow, oh)
        return detections

    # ── Pose: one batched pass over every person crop in the batch ───────────
    def _pose_pass(self, staged: list) -> None:
        if not (self._pose_enabled and self._pose and self._classifier):
            return

        per_frame: list[tuple[np.ndarray, list]] = []
        batch_cams: set[int] = set()
        for cam_id, _ts, frame, dets in staged:
            batch_cams.add(cam_id)
            persons = [d for d in dets if d.is_person and d.track_id >= 0]
            if len(persons) > self._pose_max_persons:
                # Keep the biggest boxes — nearest people, and the ones with
                # enough pixels for a usable skeleton anyway.
                persons.sort(
                    key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                    reverse=True)
                persons = persons[:self._pose_max_persons]
            boxes = [(*d.bbox, (cam_id, d.track_id)) for d in persons]
            if boxes:
                per_frame.append((frame, boxes))

        kp_by_key = self._pose.run_batch(per_frame) if per_frame else {}

        live_keys: set = set()
        for cam_id, _ts, _frame, dets in staged:
            for d in dets:
                if not (d.is_person and d.track_id >= 0):
                    continue
                key = (cam_id, d.track_id)
                live_keys.add(key)
                if key in kp_by_key:
                    d.posture = self._classifier.classify(key, kp_by_key[key])

        # Prune classifier state only for cameras we actually processed this
        # pass — a camera that was motion-gated or rate-skipped keeps its state.
        keep_other = {k for k in self._classifier.known_keys()
                      if not (isinstance(k, tuple) and k[0] in batch_cams)}
        self._classifier.flush_missing(live_keys | keep_other)

    # ── ANPR: one batched plate-detector pass over every vehicle crop ────────
    def _vehicle_pass(self, staged: list) -> None:
        """Event-triggered ANPR: the tick a vehicle track first appears is its
        one "arrival" (snapshot + plate submit + vehicle-type classify). A
        bounded number of follow-up ticks keep feeding the plate voter; after
        that (or once a confident plate exists) the track is idle — nothing is
        submitted for it again until it disappears and a different track_id
        genuinely arrives. Cameras with zero tracked vehicles this pass are
        skipped entirely: true idle, no GPU work of any kind."""
        if self.anpr is None and self.vehicle_type is None:
            return
        now = time.time()

        # Follow-up reads are paced to every Nth *detection pass* (not frame
        # tick: detection itself only runs every few ticks, so a tick-based
        # modulo could share no phase with it and never fire), so a vehicle
        # approaching the camera is sampled across its whole approach — its
        # plate is usually unreadable on the pass it first appears.
        self._vehicle_passes += 1
        every = getattr(self.anpr, "_detect_every", 1) if self.anpr is not None else 1
        cadence = self._vehicle_passes % max(1, every) == 0
        for cam_id, ts, frame, dets in staged:
            vehicles = [d for d in dets if d.is_vehicle and d.track_id >= 0]
            if not vehicles:
                continue
            reads = self.anpr.readings_for_cam(cam_id) if self.anpr is not None else {}
            stop_after = getattr(self.anpr, "stop_after_reads", 3)
            arrivals, followups = [], []
            for d in vehicles:
                key = (cam_id, d.track_id)
                r = reads.get(d.track_id)
                confident = r is not None and r.confirmed and r.reads >= stop_after
                if self._vehicle_arrival.is_new(key) or cadence:
                    state = self._vehicle_arrival.classify(key, self._tick, now, confident)
                else:
                    continue
                if state == "arrival":
                    arrivals.append(d)
                elif state == "followup" and self.anpr is not None:
                    followups.append(d)
                # state == "done" -> nothing submitted: the idle state
            native = self._native_for(cam_id, ts) if (arrivals or followups) else None
            if arrivals:
                self._fire_arrival(cam_id, frame, arrivals, now, native)
            if followups:
                self.anpr.submit_batch(
                    [(cam_id, frame, [(*d.bbox, d.track_id) for d in followups], native)],
                    self._tick, force=True)

        # Read back plate / vehicle-type results onto live Detections, and
        # drop per-track state once a vehicle is no longer tracked.
        for cam_id, _ts, _frame, dets in staged:
            live_veh = {d.track_id for d in dets
                        if d.is_vehicle and d.track_id >= 0}
            reads = self.anpr.readings_for_cam(cam_id) if self.anpr is not None else {}
            for d in dets:
                if not d.is_vehicle:
                    continue
                r = reads.get(d.track_id)
                if r is not None and r.confirmed:      # never show an unconfirmed guess
                    d.plate, d.plate_conf = r.text, r.conf
                vt = self._vehicle_types.get((cam_id, d.track_id))
                if vt is not None:
                    d.vehicle_type, d.vehicle_type_conf = vt.label, vt.conf

            dead = {t for t in reads if t not in live_veh}
            dead |= {tid for (cid, tid) in self._vehicle_types
                    if cid == cam_id and tid not in live_veh}
            for tid in dead:
                if self.anpr is not None:
                    self.anpr.flush_track((cam_id, tid))
                if self.vehicle_type is not None:
                    self.vehicle_type.flush_track((cam_id, tid))
                self._vehicle_types.pop((cam_id, tid), None)
            self._vehicle_arrival.prune(cam_id, live_veh)

    def _native_for(self, cam_id: int, ts: float):
        """The native-resolution frame captured at exactly `ts` (the working
        frame's own timestamp), or None. A newer native frame is NOT used: the
        vehicle boxes belong to this frame, and a moving vehicle's plate would
        be cropped from the wrong place."""
        if self._native_frame_fn is None:
            return None
        try:
            try:
                got = self._native_frame_fn(cam_id, ts)
            except TypeError:                       # older fn(cam_id) signature
                got = self._native_frame_fn(cam_id)
        except Exception as e:
            logger.debug("native-frame fetch failed (cam %s): %s", cam_id, e)
            return None
        if got is None:
            return None
        nts, img = got
        return img if abs(float(nts) - float(ts)) < 1e-6 else None

    def _fire_arrival(self, cam_id: int, frame: np.ndarray, arrivals: list,
                      now: float, native=None) -> None:
        """Runs once per vehicle track, the tick it first appears: files the
        dedicated ANPR-result snapshot (data/anpr_results/, ibvap/anpr_events.py),
        classifies vehicle type, and kicks off plate OCR for it."""
        # High-res snapshot: the camera's true native-resolution frame when the
        # capture layer has the matching one buffered, else the working frame.
        snap_frame = native if native is not None else frame

        fh, fw = frame.shape[:2]
        vt_items = []
        for d in arrivals:
            x1, y1, x2, y2 = d.bbox
            bw, bh = x2 - x1, y2 - y1
            padx, pady = int(bw * 0.18), int(bh * 0.18)
            px1, py1 = max(0, x1 - padx), max(0, y1 - pady)
            px2, py2 = min(fw, x2 + padx), min(fh, y2 + pady)
            crop = frame[py1:py2, px1:px2]
            if crop.size:
                vt_items.append((cam_id, d.track_id, crop, d.class_id))
            if self.anpr_results is not None:
                self.anpr_results.capture_snapshot(
                    cam_id, snap_frame, d.track_id,
                    {"detail": d.class_name}, now)

        if vt_items and self.vehicle_type is not None:
            for key, vt in self.vehicle_type.classify_batch(vt_items).items():
                self._vehicle_types[key] = vt
                if self.anpr_results is not None:
                    self.anpr_results.attach_vehicle_type(key[0], key[1], vt)

        if self.anpr is not None:
            vehicles = [(*d.bbox, d.track_id) for d in arrivals]
            self.anpr.submit_batch([(cam_id, frame, vehicles, native)], self._tick,
                                   force=True)

    # ── Weapon: one batched firearm pass over every camera frame ─────────────
    def _weapon_pass(self, staged: list) -> dict[int, list]:
        """Returns {cam_id: [WeaponHit, ...]}. Empty dict when disabled."""
        if self.weapon is None:
            return {}
        per_cam = [
            (cam_id, frame,
             [(*d.bbox, d.track_id,
               bool(d.posture is not None and d.posture.chest_aim),
               _wrists(d))
              for d in dets if d.is_person and d.track_id >= 0])
            for cam_id, _ts, frame, dets in staged
        ]
        out = self.weapon.detect_batch(per_cam, self._tick)

        # Drop weapon streak/state for person tracks no longer live on their cam.
        for cam_id, _ts, _frame, dets in staged:
            live = {d.track_id for d in dets if d.is_person and d.track_id >= 0}
            for hit in out.get(cam_id, []):
                if hit.person_track >= 0 and hit.person_track not in live:
                    self.weapon.flush_track((cam_id, hit.person_track))
        return out

    # ── Face recognition: burst-of-N capture per person arrival, then idle ───
    def _face_pass(self, staged: list) -> dict[int, list]:
        """Returns {cam_id: [FaceHit, ...]}. Empty dict when disabled.

        Mirrors _vehicle_pass's event-triggered shape, but the "arrival" here
        is a bounded BURST (ibvap/face_events.PersonArrivalTracker) rather than
        one shot + open-ended follow-ups: up to face.burst_n face-bearing
        readings are collected per person track, then the burst is decided by
        majority vote and the track goes idle — no further GPU work for it
        until it disappears and a different track_id genuinely arrives."""
        if self.face is None:
            return {}
        now = time.time()
        due_by_cam: list[tuple[int, np.ndarray, list]] = []

        # New capture attempts are paced to every Nth detection pass
        # (face.detect_every) to spread GPU cost when several people arrive at
        # once — but reading back an already-finalized match (below) always
        # runs, every pass, so a confirmed match's box/banner never flickers
        # on a tick this gate skips.
        self._face_passes += 1
        if self._face_passes % self._face_detect_every == 0:
            for cam_id, ts, frame, dets in staged:
                persons = [d for d in dets if d.is_person and d.track_id >= 0]
                if not persons:
                    continue
                due = [d for d in persons
                      if self._face_arrival.wants_capture((cam_id, d.track_id), self._tick, now)]
                if not due:
                    continue
                native = self._native_for(cam_id, ts) if self._face_use_native else None
                src = native if native is not None else frame
                due_by_cam.append((cam_id, src, [(*d.bbox, d.track_id) for d in due]))

        hits_by_cam = self.face.identify_batch(due_by_cam) if due_by_cam else {}

        for cam_id, src, persons in due_by_cam:
            hit_by_track = {h.person_track: h for h in hits_by_cam.get(cam_id, [])}
            for x1, y1, x2, y2, tid in persons:
                key = (cam_id, tid)
                hit = hit_by_track.get(tid)
                if hit is not None:
                    seq = self._face_arrival.attempts_so_far(key)
                    self.face_results.capture_snapshot(
                        cam_id, src, tid, seq,
                        {"detail": hit.matched_name or ""}, now)
                verdict = self._face_arrival.record(key, now, hit)
                if verdict is not None:
                    rec = self.face_results.finalize(cam_id, tid, verdict, now)
                    if rec is not None:
                        self._face_arrivals.append(rec)

        # Read back matched identity onto live Detections (only once a burst
        # has actually finalized to "on watchlist" — never an in-progress
        # guess), and keep the box/banner alive for hit_ttl_s between bursts
        # via fresh_hit() on tracks that weren't due this tick.
        faces_by_cam: dict[int, list] = {}
        for cam_id, _ts, _frame, dets in staged:
            out = list(hits_by_cam.get(cam_id, []))
            have = {h.person_track for h in out}
            for d in dets:
                if not (d.is_person and d.track_id >= 0) or d.track_id in have:
                    continue
                fresh = self._face_arrival.fresh_hit((cam_id, d.track_id), now)
                if fresh is not None:
                    out.append(fresh)
            faces_by_cam[cam_id] = out
            hit_by_track = {h.person_track: h for h in out}
            for d in dets:
                if not d.is_person:
                    continue
                h = hit_by_track.get(d.track_id)
                if h is not None and h.on_watchlist:
                    d.face_name, d.face_conf, d.on_watchlist = (
                        h.matched_name, h.similarity, True)

        for cam_id, _ts, _frame, dets in staged:
            live = {d.track_id for d in dets if d.is_person and d.track_id >= 0}
            self._face_arrival.prune(cam_id, live)
        return faces_by_cam

    def drain_face_arrivals(self) -> list[dict]:
        """Finalized face-burst records (matched or "unknown") since the last
        call — server.py files each to data/face_results/ and hash-chains a
        confirmed watchlist match into evidence."""
        out, self._face_arrivals = self._face_arrivals, []
        return out

    # ── Carrying tracks between detector passes ──────────────────────────────
    def _carry_forward(self, cam_id: int, ts: float, frame: np.ndarray) -> StreamResult:
        """Advance known boxes by their velocity — no GPU work at all."""
        st = self._cam(cam_id)
        age = max(0, self._tick - st.last_detect_tick)

        carried: list[Detection] = []
        h, w = frame.shape[:2]
        for d in st.detections:
            dx, dy = d.velocity[0] * age, d.velocity[1] * age
            x1, y1, x2, y2 = d.bbox
            nx1, ny1 = int(x1 + dx), int(y1 + dy)
            nx2, ny2 = int(x2 + dx), int(y2 + dy)
            # Drop tracks that have drifted off-frame rather than pinning them
            # to the edge, which would show a phantom target on the wall.
            if nx2 <= 0 or ny2 <= 0 or nx1 >= w or ny1 >= h:
                continue
            carried.append(Detection(
                d.track_id, d.class_id, d.class_name,
                (max(nx1, 0), max(ny1, 0), min(nx2, w), min(ny2, h)),
                d.confidence, d.is_person, d.is_vehicle, d.velocity,
                posture=d.posture,   # carry last known pose — dimmed in draw_skeleton
                plate=d.plate, plate_conf=d.plate_conf,
                vehicle_type=d.vehicle_type, vehicle_type_conf=d.vehicle_type_conf,
                face_name=d.face_name, face_conf=d.face_conf, on_watchlist=d.on_watchlist,
            ))

        return StreamResult(
            cam_id, ts, frame, carried, st.person_count, st.vehicle_count,
            self._annotate(frame, carried, dim=True),
            inferred=False, gated=True,
        )

    # ── Drawing ──────────────────────────────────────────────────────────────
    def _annotate(self, frame: np.ndarray, detections: list[Detection],
                  dim: bool = False, weapons: list | None = None,
                  faces: list | None = None) -> np.ndarray:
        canvas = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            if d.is_person:
                colour = (60, 220, 60)
            elif d.is_vehicle:
                colour = (240, 170, 40)
            else:
                colour = (170, 170, 170)
            if dim:                       # carried box: slightly muted
                colour = tuple(int(c * 0.72) for c in colour)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            label = f"#{d.track_id} {d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), colour, -1)
            cv2.putText(canvas, label, (x1 + 3, ty - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)
            # Skeleton overlay on persons that have pose data
            if d.is_person and d.posture is not None:
                draw_skeleton(canvas, d.posture, dim=dim)
            # Number plate + vehicle type under a vehicle box
            if d.is_vehicle and d.plate:
                pl = f"{d.plate}  {d.plate_conf:.0%}"
                if d.vehicle_type:
                    pl += f"  {d.vehicle_type}"
                (pw, ph), _ = cv2.getTextSize(pl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                py = min(y2 + ph + 6, canvas.shape[0] - 2)
                cv2.rectangle(canvas, (x1, py - ph - 5), (x1 + pw + 8, py + 2),
                              (0, 215, 255), -1)
                cv2.putText(canvas, pl, (x1 + 4, py - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 2, cv2.LINE_AA)

        # Weapon boxes: bright red + "GUN" when confirmed (held + temporal vote),
        # faint dashed-look otherwise. BGR — red is (0, 0, 255).
        for hh in (weapons or []):
            wx1, wy1, wx2, wy2 = hh.bbox
            if hh.confirmed:
                wcol, thick = (0, 0, 255), 3
                wlabel = f"GUN {hh.conf:.0%}"
            else:
                wcol, thick = (60, 60, 200), 1
                wlabel = f"gun? {hh.conf:.0%}"
            cv2.rectangle(canvas, (wx1, wy1), (wx2, wy2), wcol, thick)
            (gw, gh), _ = cv2.getTextSize(wlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            gy = max(wy1 - 4, gh + 4)
            cv2.rectangle(canvas, (wx1, gy - gh - 4), (wx1 + gw + 8, gy + 2), wcol, -1)
            cv2.putText(canvas, wlabel, (wx1 + 4, gy - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Face boxes: bright red + name when on the watchlist, faint cyan
        # "checked" tick otherwise — so an operator sees a face was looked at
        # even when it didn't match anyone.
        for fh in (faces or []):
            fx1, fy1, fx2, fy2 = fh.bbox
            if fh.on_watchlist:
                fcol, thick = (0, 0, 255), 3
                flabel = f"{fh.matched_name or '?'} {fh.similarity:.0%}"
            else:
                fcol, thick = (200, 160, 60), 1
                flabel = "face checked"
            cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), fcol, thick)
            (fw_, fh_), _ = cv2.getTextSize(flabel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            fy = max(fy1 - 4, fh_ + 4)
            cv2.rectangle(canvas, (fx1, fy - fh_ - 4), (fx1 + fw_ + 8, fy + 2), fcol, -1)
            cv2.putText(canvas, flabel, (fx1 + 4, fy - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    # ── Introspection ────────────────────────────────────────────────────────
    # A camera counts as "live" (motion is currently reaching YOLO) if the
    # motion gate passed one of its frames within this many ticks. Otherwise
    # it is "idle" — a static scene the GPU is not being spent on.
    LIVE_WITHIN_TICKS = 12

    def gate_stats(self) -> dict[int, dict]:
        return {
            cam_id: {
                "considered": st.gate.stats.considered,
                "skipped": st.gate.stats.skipped,
                "skip_pct": round(st.gate.stats.skip_pct, 1),
                "last_motion": round(st.gate.stats.last_ratio, 5),
                "infer_count": st.infer_count,
                "gated_count": st.gated_count,
                # live = motion gating is currently passing frames through
                "live": (self._tick - st.last_gate_pass_tick) <= self.LIVE_WITHIN_TICKS,
            }
            for cam_id, st in self._cams.items()
        }

    # ── Backwards-compatible single-frame path ───────────────────────────────
    def process(self, cam_id: int, timestamp: float, frame: np.ndarray) -> StreamResult:
        return self.process_batch([(cam_id, timestamp, frame)])[0]
