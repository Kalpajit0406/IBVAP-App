"""
anpr.py — Automatic Number-Plate Recognition, wired into the IBVAP pipeline.

Two stages:

  1. a YOLO plate detector (models/license_plate_detector.pt, one class
     `license_plate`) runs on each tracked *vehicle crop* — cut from the
     camera's native-resolution frame when the capture layer has the matching
     one, since a plate that is 60 px wide at 1080p is 40 px after the 720p
     working-frame resize and OCR accuracy collapses with it;
  2. fast-plate-ocr (a plate-specific ONNX recogniser) reads each plate box on a
     background worker thread, never touching the real-time budget. EasyOCR is
     the fallback backend.

Calibrated on real Indian street footage (see docs/ANPR.md, "Calibration"):

  * every plausible plate box on a vehicle is read (up to
    `max_plates_per_vehicle`), not just the most confident one — badges such as
    "NEXON" routinely out-score the real plate in the detector;
  * two-line plates (motorcycles, autos, trucks) are also read row by row;
  * the OCR's full per-character probability matrix is kept, not just its
    top guess. Each read is decoded under the Indian registration grammar
    (known state code, 1-2 digit RTO, 0-3 letter series, 4-digit number, or the
    BH series): the most probable *valid* plate, so a "4" that was 30% "0" in
    the RTO slot comes out right instead of being rejected or mis-corrected;
  * reads of one vehicle are fused by averaging those probabilities across
    frames and decoding the fusion under the same grammar, and a plate is only
    *confirmed* — shown, logged, evidenced — once `confirm_min_reads` frames
    independently decode to it and the fused confidence clears
    `confirm_min_conf`.

If the plate model or an OCR backend is missing the engine disables itself with
one warning and the rest of the pipeline is unaffected.
"""
from __future__ import annotations

import csv
import logging
import queue
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("ibvap.anpr")

# India: state/UT codes in use on registration plates (incl. legacy + BH series).
IN_STATES = frozenset(
    "AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ "
    "NL OD OR PB PY RJ SK TG TN TR TS UK UP WB".split())

# Region plate-format patterns (post-cleanup: A–Z0–9 only, no spaces).
_REGION_RE = {
    # India: SS RR L(L)(L) NNNN  e.g. WB06AB1234, DL8CAY4767, KA01A9999, or BH series 22BH1234AA
    "IN": re.compile(r"^(?:[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}|[0-9]{2}BH[0-9]{4}[A-Z]{1,2})$"),
    # Generic: 4–10 alphanumerics with at least one letter and one digit
    "XX": re.compile(r"^(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{4,10}$"),
}

# Look-alike substitutions OCR makes between glyph classes.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
             "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
_TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}


def _edit_distance(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)]


@dataclass
class PlateReading:
    text: str
    conf: float
    bbox: tuple[int, int, int, int]      # plate box in working-frame coords
    cam_id: int
    track_id: int
    ts: float
    valid: bool = False                  # matched the region format
    reads: int = 1                       # frames whose reading equals `text`
    confirmed: bool = False              # enough agreement to act on


@dataclass
class _Stats:
    plates_detected: int = 0             # plate boxes the YOLO stage queued for OCR
    ocr_runs: int = 0
    valid_reads: int = 0
    plates_logged: int = 0
    queue_drops: int = 0


class AnprEngine:
    def __init__(self, cfg: dict, device: str = "0") -> None:
        self.available = False
        self._cfg = cfg or {}
        self._device = device
        c = self._cfg
        self._region = str(c.get("region", "IN")).upper()
        self._yolo_conf = float(c.get("yolo_conf", 0.25))
        self._ocr_min_conf = float(c.get("ocr_min_conf", 0.20))
        self._min_area = int(c.get("min_plate_area", 300))          # working-frame px²
        self._min_plate_w = int(c.get("min_plate_width", 36))       # native px; narrower = unreadable
        self._detect_every = max(1, int(c.get("plate_detect_every", 3)))
        self._cooldown = float(c.get("cooldown_s", 8.0))
        self._plate_imgsz = int(c.get("plate_imgsz", 320))
        self._max_boxes = max(1, int(c.get("max_plates_per_vehicle", 2)))
        self._plate_pad_x = float(c.get("plate_pad_x", c.get("plate_pad", 0.10)))
        self._plate_pad_y = float(c.get("plate_pad_y", c.get("plate_pad", 0.08)))
        self._read_min_conf = float(c.get("read_min_conf", 0.45))
        self._confirm_conf = float(c.get("confirm_min_conf", 0.50))
        self._confirm_share = float(c.get("confirm_min_share", 0.50))
        self._drop_max = float(c.get("edge_drop_max_conf", 0.60))
        self._gray_retry = bool(c.get("ocr_gray_retry", False))
        self._dup_window = float(c.get("near_duplicate_window_s", 30.0))
        self._two_line_ratio = float(c.get("two_line_ratio", 0.42))
        self._ocr_color = str(c.get("ocr_color", "bgr")).lower()
        self._confirm_reads = max(1, int(c.get("confirm_min_reads", 2)))
        self._single_read_conf = float(c.get("single_read_min_char_conf", 0.97))
        self._max_coerce = int(c.get("max_corrections", 2))
        self.stop_after_reads = max(self._confirm_reads,
                                    int(c.get("stop_after_confirmed_reads", 3)))
        self._log_unconfirmed = bool(c.get("log_unconfirmed", False))
        n_workers = max(1, int(c.get("workers", 1)))

        # OCR backend: "fast_plate" (fast-plate-ocr, plate-specific ONNX) falling
        # back to "easyocr" (generic English) when it isn't installed.
        self._ocr_backend = str(c.get("ocr_backend", "fast_plate")).lower()
        self._ocr_model = str(c.get("ocr_model", "cct-s-v2-global-model"))
        self._vote_min = max(1, int(c.get("vote_min_reads", 3)))
        self._vote_max = max(self._vote_min, int(c.get("vote_max_reads", 12)))
        # (cam,tid) -> [(text, weight, ts, seq_probs L×A | None), …]  — grammar-valid reads only
        self._reads: dict[tuple, list] = {}

        self.stats = _Stats()
        # (cam_id, track_id) → current reading. The composite key keeps vehicle 1
        # on camera A distinct from vehicle 1 on camera B.
        self._plates: dict[tuple, PlateReading] = {}
        self._plates_lock = threading.Lock()
        self._reads_lock = threading.Lock()
        self._events: list[PlateReading] = []
        self._events_lock = threading.Lock()
        self._last_logged: dict[str, float] = {}       # plate text → last CSV time
        self._logged_for_track: dict[tuple, str] = {}  # (cam,tid) → text last logged
        # cam_id → [(text, track_id, ts)] recently confirmed, for near-duplicate suppression
        self._recent_confirmed: dict[int, list] = defaultdict(list)
        self._q: queue.Queue = queue.Queue(maxsize=max(4, int(c.get("queue_size", 32))))
        self._stop = threading.Event()

        weights = c.get("weights", "models/license_plate_detector.pt")
        if not Path(weights).exists():
            logger.warning("ANPR disabled — plate model not found at %s "
                           "(see docs/ANPR.md)", weights)
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            logger.warning("ANPR disabled — %s", e)
            return
        try:
            self._plate_model = YOLO(str(weights))
        except Exception as e:
            logger.error("ANPR disabled — plate model failed to load (%s)", e)
            return

        # ── OCR backend ────────────────────────────────────────────────────
        gpu = str(device) != "cpu"
        self._fp = None          # fast-plate-ocr LicensePlateRecognizer, or None
        self._reader = None      # easyocr.Reader, or None
        if self._ocr_backend == "fast_plate":
            model_id = self._ocr_model
            is_path = "/" in model_id or "\\" in model_id
            # Pin the ONNX Runtime providers: CUDA (or CPU) only. The TensorRT EP
            # that onnxruntime-gpu registers first needs nvinfer_*.dll on PATH and
            # is pointless for a ~5 MB model — skipping it avoids a noisy EP error.
            dev = str(c.get("ocr_device", "auto")).lower()
            if dev == "auto":
                dev = "cuda" if gpu else "cpu"
            prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if dev == "cuda" else ["CPUExecutionProvider"])
            try:
                from fast_plate_ocr import LicensePlateRecognizer
                if is_path and Path(model_id).is_dir():
                    onnx_files = sorted(Path(model_id).glob("*.onnx"))
                    cfg_files = (sorted(Path(model_id).glob("*plate_config*.yaml"))
                                 or sorted(Path(model_id).glob("*.yaml")))
                    if onnx_files and cfg_files:
                        try:
                            self._fp = LicensePlateRecognizer(
                                onnx_model_path=str(onnx_files[0]),
                                plate_config_path=str(cfg_files[0]),
                                providers=prov)
                        except TypeError:
                            self._fp = LicensePlateRecognizer(
                                onnx_model_path=str(onnx_files[0]),
                                plate_config_path=str(cfg_files[0]), device=dev)
                        logger.info("ANPR OCR backend: fast-plate-ocr (local %s, %s)",
                                    model_id, prov[0])
                    else:
                        logger.info("ANPR: %s has no model.onnx + config — "
                                    "using bundled cct-s-v2-global-model", model_id)
                if self._fp is None:
                    if is_path:
                        model_id = "cct-s-v2-global-model"
                    try:
                        self._fp = LicensePlateRecognizer(model_id, providers=prov)
                    except TypeError:            # older fast-plate-ocr: no providers=
                        self._fp = LicensePlateRecognizer(model_id, device=dev)
                    self._ocr_model = model_id
                    logger.info("ANPR OCR backend: fast-plate-ocr (%s, %s)", model_id, prov[0])
            except Exception as e:
                logger.warning("ANPR: fast-plate-ocr unavailable (%s) — "
                               "falling back to EasyOCR", e)
        if self._fp is None:
            try:
                import easyocr
                logger.info("ANPR: loading EasyOCR (gpu=%s) — first run downloads "
                            "~64 MB of OCR models", gpu)
                self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
                self._ocr_backend = "easyocr"
            except ImportError:
                logger.warning("ANPR disabled — no OCR backend. Run one of: "
                               "pip install \"fast-plate-ocr[onnx-gpu]\"  /  pip install easyocr")
                return
            except Exception as e:
                logger.error("ANPR disabled — EasyOCR failed to initialise (%s)", e)
                return

        self._csv_path = Path("data/plates/plates.csv")
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["timestamp", "cam_id", "track_id", "plate", "confidence",
                     "valid", "crop"])

        self._workers = [
            threading.Thread(target=self._ocr_worker, daemon=True,
                             name=f"anpr-ocr-{i}")
            for i in range(n_workers)
        ]
        for w in self._workers:
            w.start()
        self.available = True
        logger.info("ANPR ready — plate model %s, OCR %s, region %s, confirm>=%d reads, "
                    "<=%d plate boxes/vehicle, %d worker(s)", Path(weights).name,
                    self._ocr_backend, self._region, self._confirm_reads,
                    self._max_boxes, n_workers)

    # ── stage 1: plate detection on vehicle crops (caller's thread) ──────────
    def submit(self, cam_id: int, frame: np.ndarray,
               vehicles: list[tuple[int, int, int, int, int]], tick: int,
               native: Optional[np.ndarray] = None, force: bool = False) -> None:
        """Single-camera wrapper around :meth:`submit_batch`."""
        self.submit_batch([(cam_id, frame, vehicles, native)], tick, force=force)

    def submit_batch(self, per_cam, tick: int, force: bool = False) -> None:
        """
        Run the plate detector once for every vehicle crop in a detection
        batch — one forward pass across all cameras, not one per camera.

        per_cam: iterable of (cam_id, frame_bgr, vehicles[, native_bgr]) where
                 vehicles is a list of (x1, y1, x2, y2, track_id) in `frame`
                 coordinates and native_bgr (optional) is the same moment at
                 the camera's full resolution.
        force:   skip the `plate_detect_every` tick gate — for event-driven
                 callers (vehicle arrival / follow-up) that already pace
                 themselves; gating those would silently drop them.
        """
        if not self.available:
            return
        if not force and tick % self._detect_every != 0:
            return
        now = time.time()
        crops, meta = [], []       # meta: (cam_id, src, tid, ox, oy, sx, sy)
        for item in per_cam:
            cam_id, frame, vehicles = item[0], item[1], item[2]
            native = item[3] if len(item) > 3 else None
            if not vehicles or frame is None:
                continue
            fh, fw = frame.shape[:2]
            src = native if native is not None else frame
            sh, sw = src.shape[:2]
            sx, sy = sw / fw, sh / fh
            for (x1, y1, x2, y2, tid) in vehicles:
                with self._plates_lock:
                    r = self._plates.get((cam_id, tid))
                if (r is not None and r.confirmed and r.reads >= self.stop_after_reads
                        and (now - r.ts) < self._cooldown):
                    continue                      # plate already settled
                bw, bh = (x2 - x1) * sx, (y2 - y1) * sy
                pad_x, pad_y = max(8.0, bw * 0.06), max(8.0, bh * 0.06)
                px1 = max(0, int(x1 * sx - pad_x)); py1 = max(0, int(y1 * sy - pad_y))
                px2 = min(sw, int(x2 * sx + pad_x)); py2 = min(sh, int(y2 * sy + pad_y))
                if px2 - px1 < 24 or py2 - py1 < 24:
                    continue
                crops.append(src[py1:py2, px1:px2])
                meta.append((cam_id, src, tid, px1, py1, sx, sy))

        if not crops:
            return
        try:
            outs = self._plate_model.predict(
                crops, conf=self._yolo_conf, imgsz=self._plate_imgsz,
                device=self._device, verbose=False)
        except Exception as e:
            logger.debug("ANPR plate-detect failed: %s", e)
            return

        for (cam_id, src, tid, ox, oy, sx, sy), out in zip(meta, outs):
            sh, sw = src.shape[:2]
            boxes = sorted(((float(b.conf[0]), [int(v) for v in b.xyxy[0]]) for b in out.boxes),
                           key=lambda t: -t[0])
            taken = 0
            for conf, (bx1, by1, bx2, by2) in boxes:
                pw, ph = bx2 - bx1, by2 - by1
                if pw < self._min_plate_w or ph < 8:
                    continue
                if pw * ph < self._min_area * sx * sy:
                    continue
                if not (0.9 <= pw / max(ph, 1) <= 8.0):          # not plate-shaped
                    continue
                padx, pady = int(round(pw * self._plate_pad_x)), int(round(ph * self._plate_pad_y))
                fx1, fy1 = bx1 + ox, by1 + oy
                fx2, fy2 = bx2 + ox, by2 + oy
                plate_img = src[max(0, fy1 - pady):min(sh, fy2 + pady),
                                max(0, fx1 - padx):min(sw, fx2 + padx)].copy()
                if plate_img.size == 0:
                    continue
                bbox = (int(fx1 / sx), int(fy1 / sy), int(fx2 / sx), int(fy2 / sy))
                try:
                    self._q.put_nowait((cam_id, tid, plate_img, bbox, now))
                    self.stats.plates_detected += 1
                except queue.Full:
                    self.stats.queue_drops += 1
                taken += 1
                if taken >= self._max_boxes:
                    break

    # ── stage 2: OCR (worker thread) ───────────────────────────────────────
    @staticmethod
    def _preprocess(roi: np.ndarray):
        h, w = roi.shape[:2]
        if h < 80 or w < 160:
            s = max(80.0 / max(h, 1), 160.0 / max(w, 1), 2.0)
            roi = cv2.resize(roi, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        return roi, gray, enhanced

    @staticmethod
    def _clean(raw: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", raw).upper()

    # Kept for callers of the old translate tables.
    _TO_DIGIT = str.maketrans("".join(_TO_DIGIT), "".join(_TO_DIGIT.values()))
    _TO_ALPHA = str.maketrans("".join(_TO_ALPHA), "".join(_TO_ALPHA.values()))

    @staticmethod
    def _segment_cost(t: str, pattern: str) -> tuple[int, str] | None:
        """Cost of forcing `t` into `pattern` ('A' letter / 'D' digit per char)
        using only look-alike swaps; None when some character can't fit."""
        out, cost = [], 0
        for ch, want in zip(t, pattern):
            if want == "A":
                if ch.isalpha():
                    out.append(ch)
                elif ch in _TO_ALPHA:
                    out.append(_TO_ALPHA[ch]); cost += 1
                else:
                    return None
            else:
                if ch.isdigit():
                    out.append(ch)
                elif ch in _TO_DIGIT:
                    out.append(_TO_DIGIT[ch]); cost += 1
                else:
                    return None
        return cost, "".join(out)

    def _coerce_in(self, t: str) -> tuple[str, bool]:
        """Force a reading into a valid Indian registration with the fewest
        look-alike corrections. Tries every segmentation (1-2 digit RTO, 0-3
        letter series, 4-digit number, and the BH series), and tolerates one
        stray character at either end (a plate frame / 'IND' strip). Returns
        (text, matched)."""
        if not t:
            return t, False
        if _REGION_RE["IN"].match(t) and (t[:2] in IN_STATES or t[2:4] == "BH"):
            return t, True
        cands = [(t, 0)]
        if t.startswith("IND"):
            cands.append((t[3:], 0))
        if len(t) >= 9:
            cands += [(t[1:], 1), (t[:-1], 1)]
        best: tuple[int, str] | None = None
        for s, extra in cands:
            n = len(s)
            patterns = []
            for rto in (1, 2):
                for series in (0, 1, 2, 3):
                    if 2 + rto + series + 4 == n:
                        patterns.append("AA" + "D" * rto + "A" * series + "DDDD")
            for suffix in (1, 2):
                if 2 + 2 + 4 + suffix == n:
                    patterns.append("DDAADDDD" + "A" * suffix)       # BH series
            for pat in patterns:
                r = self._segment_cost(s, pat)
                if r is None:
                    continue
                cost, out = r
                cost += extra
                if pat.startswith("DDAA"):
                    if out[2:4] != "BH":
                        continue
                elif out[:2] not in IN_STATES:
                    continue
                if cost <= self._max_coerce and (best is None or cost < best[0]):
                    best = (cost, out)
        if best is None:
            return t, False
        return best[1], True

    _ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # ── fast-plate-ocr: raw per-slot probabilities ─────────────────────────
    def _ocr_probs(self, imgs: list) -> list[np.ndarray]:
        """Per-image character-probability sequences, shape (L, alphabet): the
        model's non-pad slots in order. Falls back to one-hot sequences built
        from `run()` text when the recogniser internals aren't reachable."""
        if self._ocr_color == "rgb":
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) if i.ndim == 3 else i for i in imgs]
        fp = self._fp
        try:
            from fast_plate_ocr.inference import plate_recognizer as _pr
            x = _pr.preprocess_image(_pr._load_image_from_source(list(imgs), fp.config))
            out = fp.model.run([fp.plate_output_name], {"input": x})[0]
            A = len(fp.config.alphabet)
            out = np.asarray(out, dtype=np.float32).reshape(len(imgs), -1, A)
            pad = fp.config.alphabet.index(fp.config.pad_char)
            seqs = []
            for m in out:
                am = m.argmax(axis=1)
                L = len(am)
                while L and am[L - 1] == pad:
                    L -= 1
                seqs.append(m[:L])
            return seqs
        except Exception as e:                        # pragma: no cover - old library
            logger.debug("raw OCR probabilities unavailable (%s) — one-hot fallback", e)
            alphabet = getattr(getattr(fp, "config", None), "alphabet",
                               "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_")
            seqs = []
            for o in fp.run(list(imgs)):
                t = self._clean(str(getattr(o, "plate", o)))
                m = np.full((len(t), len(alphabet)), 1e-3, np.float32)
                for i, ch in enumerate(t):
                    if ch in alphabet:
                        m[i, alphabet.index(ch)] = 0.9
                seqs.append(m)
            return seqs

    def _alphabet_index(self):
        if not hasattr(self, "_abc"):
            alphabet = getattr(getattr(self._fp, "config", None), "alphabet",
                               "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_")
            letters = [i for i, ch in enumerate(alphabet) if ch.isalpha()]
            digits = [i for i, ch in enumerate(alphabet) if ch.isdigit()]
            states = [(alphabet.index(st[0]), alphabet.index(st[1])) for st in sorted(IN_STATES)
                      if st[0] in alphabet and st[1] in alphabet]
            self._abc = (alphabet, np.array(letters), np.array(digits), states)
        return self._abc

    def _patterns(self, n: int) -> list[str]:
        pats = []
        for rto in (1, 2):
            for series in (0, 1, 2, 3):
                if 2 + rto + series + 4 == n:
                    pats.append("SS" + "D" * rto + "A" * series + "DDDD")
        for suffix in (1, 2):
            if 8 + suffix == n:
                pats.append("DDBHDDDD" + "A" * suffix)
        return pats

    def _decode_seq(self, seq: np.ndarray, allow_drop: bool = True):
        """Most probable grammar-valid plate for a probability sequence.
        Returns (text, geometric-mean char prob, min char prob, aligned_seq) or
        None; `aligned_seq` is the slice the text was decoded from, so reads
        can be fused slot-for-slot. One stray character at either end
        (plate-frame edge, "IND" strip) may be dropped at a 10% penalty — only
        when the OCR itself was unsure of it, otherwise a real plate
        character would be thrown away to fit a shorter pattern."""
        if self._region != "IN":
            alphabet = self._alphabet_index()[0]
            if not len(seq):
                return None
            idx = seq.argmax(axis=1)
            text = "".join(alphabet[i] for i in idx)
            probs = seq[np.arange(len(seq)), idx]
            if not _REGION_RE["XX"].match(text):
                return None
            return (text, float(np.exp(np.log(np.maximum(probs, 1e-6)).mean())),
                    float(probs.min()), seq)
        alphabet, letters, digits, states = self._alphabet_index()
        best = None
        views = [(seq, 1.0)]
        if allow_drop and len(seq) >= 9:
            if float(seq[0].max()) < self._drop_max:
                views.append((seq[1:], 0.9))
            if float(seq[-1].max()) < self._drop_max:
                views.append((seq[:-1], 0.9))
        for view, penalty in views:
            n = len(view)
            for pat in self._patterns(n):
                chars, probs = [], []
                if pat[0] == "S":
                    pairs = [(view[0][a] * view[1][b], a, b) for a, b in states]
                    pv, a, b = max(pairs)
                    chars += [alphabet[a], alphabet[b]]
                    probs += [view[0][a], view[1][b]]
                    rest, start = pat[2:], 2
                else:
                    rest, start = pat, 0
                for i, want in enumerate(rest, start):
                    col = view[i]
                    if want == "A":
                        k = letters[col[letters].argmax()]
                    elif want == "D":
                        k = digits[col[digits].argmax()]
                    else:                                   # literal (BH series)
                        k = alphabet.index(want)
                    chars.append(alphabet[k])
                    probs.append(col[k])
                pr = np.maximum(np.asarray(probs, dtype=np.float64), 1e-6)
                geo = float(np.exp(np.log(pr).mean())) * penalty
                cand = ("".join(chars), geo, float(pr.min()) * penalty, view)
                if best is None or cand[1] > best[1]:
                    best = cand
        return best

    def _read_views(self, roi: np.ndarray, gray: bool = False) -> list[tuple[str, np.ndarray]]:
        """Probability sequences for a crop: whole plate, and — when the crop is
        tall enough to be a two-line plate — its two rows joined."""
        img = roi
        if gray:
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        two_line = h / max(w, 1) >= self._two_line_ratio
        imgs = [img] + ([img[: int(h * 0.56)], img[int(h * 0.44):]] if two_line else [])
        seqs = self._ocr_probs(imgs)
        views = [("whole" + ("/gray" if gray else ""), seqs[0])]
        if two_line:
            views.append(("split" + ("/gray" if gray else ""), np.concatenate([seqs[1], seqs[2]], axis=0)))
        return views

    def _candidates(self, roi: np.ndarray) -> list[tuple[str, float, float]]:
        """[(raw_text, decoded_conf, decoded_min)] for inspection/tests."""
        return [(c[0], c[2], c[3]) for c in self._decoded_candidates(roi)]

    def _decoded_candidates(self, roi: np.ndarray) -> list[tuple]:
        """[(raw_text, decoded_text|None, conf, min, seq, view)] for one crop."""
        alphabet = self._alphabet_index()[0]
        out = []
        views = self._read_views(roi)
        decoded = [(v, sq, self._decode_seq(sq)) for v, sq in views]
        if self._gray_retry and max((d[1] for _, _, d in decoded if d), default=0.0) < 0.6:
            decoded += [(v, sq, self._decode_seq(sq)) for v, sq in self._read_views(roi, gray=True)]
        for view, sq, d in decoded:
            raw = "".join(alphabet[i] for i in sq.argmax(axis=1)) if len(sq) else ""
            if d is None:
                out.append((raw, None, 0.0, 0.0, sq, view))
            else:
                out.append((raw, d[0], d[1], d[2], d[3], view))
        return out

    def _best_candidate(self, cands):
        """Pick the crop's reading: the most confident grammar-valid decode.
        `cands` are _decoded_candidates tuples. Returns
        (text, conf, min_conf, valid, seq)."""
        valid = [c for c in cands if c[1] is not None]
        if not valid:
            raw = max(cands, key=lambda c: len(c[0]))[0] if cands else ""
            return raw, 0.0, 0.0, False, None
        c = max(valid, key=lambda c: c[2])
        return c[1], c[2], c[3], True, c[4]

    def _read_fast_plate(self, roi) -> tuple[str, float]:
        """Best single reading for a crop (compatibility helper)."""
        text, conf, _mn, _valid, _seq = self._best_candidate(self._decoded_candidates(roi))
        return text, conf

    def _read_easyocr(self, roi) -> tuple[str, float]:
        roi_s, gray, enh = self._preprocess(roi)
        res = (self._reader.readtext(enh, allowlist=self._ALLOW)
               or self._reader.readtext(gray, allowlist=self._ALLOW)
               or self._reader.readtext(roi_s))
        parts, confs = [], []
        for item in res:
            c = self._clean(item[1])
            if len(c) >= 2:
                parts.append(c)
                confs.append(float(item[2]))
        if not parts:
            return "", 0.0
        return "".join(parts), sum(confs) / len(confs)

    @staticmethod
    def _vote(reads: list) -> tuple[str, float]:
        """Confidence-weighted positional majority across the readings of the
        modal length (text-level; used when no probability sequences exist,
        e.g. the EasyOCR backend). `reads` = [(text, weight, ts[, seq]), …].
        Returns (voted_text, mean per-position weighted agreement 0..1)."""
        texts = [(r[0], max(float(r[1]), 1e-3)) for r in reads if r[0]]
        if not texts:
            return "", 0.0
        by_len: dict[int, float] = defaultdict(float)
        for t, w in texts:
            by_len[len(t)] += w
        L = max(by_len.items(), key=lambda kv: (kv[1], kv[0]))[0]
        same = [(t, w) for t, w in texts if len(t) == L]
        total = sum(w for _, w in same)
        out, agree = [], []
        for i in range(L):
            acc: dict[str, float] = defaultdict(float)
            for t, w in same:
                acc[t[i]] += w
            ch, wt = max(acc.items(), key=lambda kv: kv[1])
            out.append(ch)
            agree.append(wt / total)
        return "".join(out), sum(agree) / len(agree)

    def _fuse(self, reads: list) -> tuple[str, float]:
        """Fuse a track's reads: average the probability sequences of the
        dominant length (weighted by each read's own decode confidence) and
        decode the result under the plate grammar. Text vote when no
        sequences are available."""
        seq_reads = [r for r in reads if len(r) > 3 and r[3] is not None]
        if not seq_reads:
            voted, agree = self._vote(reads)
            if self._region == "IN":
                voted, ok = self._coerce_in(voted)
                if not ok:
                    return "", 0.0
            return voted, agree
        by_len: dict[int, float] = defaultdict(float)
        for r in seq_reads:
            by_len[len(r[3])] += r[1]
        L = max(by_len.items(), key=lambda kv: (kv[1], kv[0]))[0]
        group = [r for r in seq_reads if len(r[3]) == L]
        wsum = sum(r[1] for r in group)
        fused = sum(r[3] * r[1] for r in group) / max(wsum, 1e-9)
        d = self._decode_seq(fused, allow_drop=False)
        if d is None:
            return "", 0.0
        # A length group holding only part of the evidence can't be fully trusted.
        share = wsum / max(sum(r[1] for r in seq_reads), 1e-9)
        return d[0], d[1] * (0.5 + 0.5 * share)

    def _near_duplicate(self, cam_id: int, tid: int, text: str, now: float) -> tuple[str, bool]:
        """A vehicle that ByteTrack drops and re-acquires gets a new track id,
        and the second track may settle one character off (DL8CAY4777 vs
        DL8CAY4767). A confirmation one edit away from a plate another track
        on this camera confirmed within `near_duplicate_window_s` is taken as
        that same vehicle: it keeps the earlier plate instead of reporting a
        new, wrong one."""
        recent = self._recent_confirmed[cam_id]
        recent[:] = [r for r in recent if now - r[2] <= self._dup_window]
        for other, other_tid, _ts in recent:
            if other_tid != tid and other != text and _edit_distance(other, text) <= 1:
                return other, True
        if not any(r[0] == text and r[1] == tid for r in recent):
            recent.append((text, tid, now))
        return text, True

    def _ocr_worker(self) -> None:
        while not self._stop.is_set():
            try:
                cam_id, tid, roi, bbox, ts = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process_crop(cam_id, tid, roi, bbox)
            except Exception as e:
                logger.debug("ANPR OCR error: %s", e)
            finally:
                self._q.task_done()

    def _process_crop(self, cam_id: int, tid: int, roi, bbox) -> None:
        # `support` (below) does double duty: it decides whether THIS reading
        # confirms, and — via PlateReading.reads feeding detector.py's
        # `confident = r.confirmed and r.reads >= stop_after_reads` gate —
        # whether the vehicle-arrival tracker stops asking for more follow-up
        # crops. Loosening it to near-match (edit distance <= 1) instead of
        # exact text equality was tried on real footage and reverted: it
        # confirms genuinely correct-but-never-repeated fusion answers faster
        # (good), but by the SAME mechanism it also locks onto an early wrong
        # near-match blend faster, on tracks where more follow-up reads would
        # have surfaced 3 clean EXACT repeats of the right plate. Exact-match
        # support is slower to confirm but effectively keeps the follow-up
        # window open longer, which on real, noisy street footage measurably
        # won more often across a track's whole follow-up budget than it lost
        # by understating support on well-fused-but-never-repeated answers. If
        # revisiting this, decouple "confirmed enough to display" from
        # "settled enough to stop collecting more reads" into two thresholds
        # instead of reusing one `support` value for both.
        self.stats.ocr_runs += 1
        if self._fp is not None:
            text, conf, mn, valid, seq = self._best_candidate(self._decoded_candidates(roi))
        else:
            raw, conf = self._read_easyocr(roi)
            text, valid = (self._coerce_in(raw) if self._region == "IN"
                           else (raw, bool(_REGION_RE["XX"].match(raw))))
            mn, seq = conf, None
        if not text or not valid or conf < self._read_min_conf:
            return                          # only confident, grammar-valid reads count
        self.stats.valid_reads += 1
        key = (cam_id, tid)
        now = time.time()
        with self._reads_lock:
            buf = self._reads.setdefault(key, [])
            buf.append((text, conf, now, seq))
            del buf[:-self._vote_max]
            snapshot = list(buf)

        fused_text, fused_conf = self._fuse(snapshot)
        if not fused_text:
            return
        support = sum(1 for r in snapshot if r[0] == fused_text)
        share = support / len(snapshot)
        confirmed = ((support >= self._confirm_reads and fused_conf >= self._confirm_conf
                      and share >= self._confirm_share) or
                     (len(snapshot) == 1 and mn >= self._single_read_conf))
        if confirmed:
            fused_text, confirmed = self._near_duplicate(cam_id, tid, fused_text, now)

        pr = PlateReading(fused_text, round(fused_conf, 3), tuple(bbox), cam_id, tid, now, True,
                          reads=support, confirmed=confirmed)
        with self._plates_lock:
            prev = self._plates.get(key)
            if (prev is None or not prev.confirmed or confirmed and support >= prev.reads
                    or (now - prev.ts) > 4.0):
                self._plates[key] = pr
        self._maybe_log(pr, roi)

    def _maybe_log(self, pr: PlateReading, crop=None) -> None:
        if not pr.confirmed and not self._log_unconfirmed:
            return
        if pr.conf < self._ocr_min_conf:
            return
        key = (pr.cam_id, pr.track_id)
        if self._logged_for_track.get(key) == pr.text:
            return
        now = time.time()
        if now - self._last_logged.get(pr.text, 0.0) < self._cooldown:
            self._logged_for_track[key] = pr.text
            return
        self._last_logged[pr.text] = now
        self._logged_for_track[key] = pr.text
        self.stats.plates_logged += 1

        crop_path = ""
        try:
            crop_dir = self._csv_path.parent
            safe = re.sub(r"[^A-Z0-9]", "_", pr.text) or "UNKNOWN"
            crop_path = str(crop_dir / f"{datetime.now():%Y%m%d_%H%M%S}_"
                                       f"cam{pr.cam_id}_{safe}.jpg")
            if crop is not None and getattr(crop, "size", 0):
                cv2.imwrite(crop_path, crop)
        except Exception:
            pass
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [datetime.now().isoformat(timespec="seconds"), pr.cam_id,
                     pr.track_id, pr.text, f"{pr.conf:.2f}", int(pr.valid),
                     crop_path])
        except Exception as e:
            logger.debug("ANPR CSV write failed: %s", e)

        with self._events_lock:
            self._events.append(pr)
        logger.info("PLATE  CAM-%02d  %s  (%.0f%%, %d reads)", pr.cam_id, pr.text,
                    pr.conf * 100, pr.reads)

    # ── consumer interface (caller's thread) ───────────────────────────────
    def readings_for_cam(self, cam_id: int, max_age: float = 5.0,
                         confirmed_only: bool = False) -> dict[int, PlateReading]:
        """{track_id: reading} for one camera (keys are bare track ids)."""
        now = time.time()
        with self._plates_lock:
            return {k[1]: r for k, r in self._plates.items()
                    if k[0] == cam_id and (now - r.ts) < max_age
                    and (r.confirmed or not confirmed_only)}

    def flush_track(self, key) -> None:
        """`key` is the (cam_id, track_id) tuple used by submit_batch. The
        camera's recently-confirmed plates are kept (for near-duplicate
        suppression) until they age out."""
        with self._plates_lock:
            self._plates.pop(key, None)
        with self._reads_lock:
            self._reads.pop(key, None)
        self._logged_for_track.pop(key, None)

    def drain_events(self) -> list[PlateReading]:
        with self._events_lock:
            out, self._events = self._events, []
        return out

    def status(self) -> dict:
        with self._plates_lock:
            active = [
                {"cam": r.cam_id, "track": k[1], "plate": r.text,
                 "conf": round(r.conf, 2), "valid": r.valid,
                 "reads": r.reads, "confirmed": r.confirmed}
                for k, r in sorted(self._plates.items())
                if (time.time() - r.ts) < 5.0 and r.confirmed
            ]
        return {
            "enabled": self.available,
            "region": self._region,
            "ocr_backend": getattr(self, "_ocr_backend", "?"),
            "plates_detected": self.stats.plates_detected,
            "ocr_runs": self.stats.ocr_runs,
            "valid_reads": self.stats.valid_reads,
            "plates_logged": self.stats.plates_logged,
            "queue_drops": self.stats.queue_drops,
            "active": active,
        }

    def stop(self) -> None:
        self._stop.set()
