"""
face.py — face detection + recognition against a small curated "watchlist"
gallery, wired into the IBVAP pipeline (`ibvap/detector.py` `_face_pass`).

Two-stage, via insightface's `buffalo_l` pack (RetinaFace detector `det_10g.onnx`
+ ArcFace R50 recognizer `w600k_r50.onnx`, both under `models/face/`, committed
to the repo — see docs/FACE_RECOGNITION.md):

  1. RetinaFace finds faces in each tracked *person crop* (padded — a tight
     person box often clips the top of the head/chin) — far cheaper and higher
     recall than scanning the whole frame for a small/distant face;
  2. ArcFace embeds the best face in that crop into a 512-d vector, already
     L2-normalized, and it is matched against every enrolled identity in the
     gallery by cosine similarity.

This is deliberately the SAME shape as an attendance-kiosk face check (detect
-> embed -> nearest-neighbour), run against a WATCHLIST instead of an
allow-list: matching one of the gallery's identities does not grant anything,
it raises a Critical "Criminal spotted" alert (`ibvap/risk_engine.py`).

A single frame's match is not trusted on its own — `ibvap/face_events.py`'s
`PersonArrivalTracker` collects up to `face.burst_n` readings per person
arrival and only calls it "on watchlist" once `face.vote_min` of them agree on
the same identity, each past `face.match_threshold`.

If the gallery, the two `.onnx` weights, or the `insightface` package are
missing the engine disables itself with one warning and the rest of the
pipeline is unaffected (same self-disabling shape as `WeaponDetector`/
`AnprEngine`), so this module can ship and run before a gallery is enrolled.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("ibvap.face")


def _unscale(box, sx: float, sy: float) -> tuple[int, int, int, int]:
    """A box measured on the detection frame, expressed back in the working
    frame's coordinates — callers (overlay, risk engine, console) only ever
    speak working-frame pixels."""
    x1, y1, x2, y2 = box
    return (int(x1 / sx), int(y1 / sy), int(x2 / sx), int(y2 / sy))


@dataclass
class FaceHit:
    person_track: int
    bbox: tuple[int, int, int, int]      # face box, full-frame pixels
    det_score: float
    matched_id: Optional[str]            # gallery identity id, or None
    matched_name: Optional[str]
    similarity: float                    # best cosine similarity vs. the gallery
    on_watchlist: bool
    trusted: bool = True                 # det_score >= det_score_min — may vote on identity
    source: str = "face"                 # "face" = detected; "head_box" = geometric guess
    quality: float = 0.0                 # 0..1 capture quality, picks the best of a burst
    face_crop: Optional[np.ndarray] = None    # BGR face crop; never serialised
    person_crop: Optional[np.ndarray] = None  # BGR whole-person crop, for context                   # similarity >= face.match_threshold


class FaceEngine:
    """One batched face-detect+recognize pass, called only for person tracks
    still inside their capture burst (see `Detector._face_pass`)."""

    def __init__(self, cfg: dict, device: str = "0", max_batch: int = 8) -> None:
        self.available = False
        self._cfg = cfg or {}
        c = self._cfg
        self._threshold = float(c.get("match_threshold", 0.38))
        self._det_score_min = float(c.get("det_score_min", 0.55))
        self._det_size = int(c.get("det_size", 320))
        self._max_persons = max(1, int(c.get("max_persons_per_pass", 8)))
        self._pad = float(c.get("crop_pad_frac", 0.25))
        # A second, much lower bar than det_score_min: clears the *snapshot*,
        # never the identity vote. Blurry/side-on/backlit faces land here.
        self._capture_score_min = float(c.get("capture_score_min", 0.20))
        self._always_capture = bool(c.get("always_capture", True))
        self._head_box_fallback = bool(c.get("head_box_fallback", True))
        self._face_pad = float(c.get("face_crop_pad_frac", 0.35))
        self._save_person_crop = bool(c.get("save_person_crop", True))
        qw = c.get("quality_weights") or {}
        self._qw = (float(qw.get("det", 0.35)), float(qw.get("area", 0.30)),
                    float(qw.get("sharp", 0.25)), float(qw.get("frontal", 0.10)))
        self._identities: list[tuple[str, str, np.ndarray]] = []

        # insightface's FaceAnalysis resolves weights by scanning
        # <model_root>/models/<pack_name>/*.onnx — so the two committed ONNX
        # files live at models/face/models/ibvap_face/*.onnx, and `model_root`
        # ("models/face") + `pack_name` ("ibvap_face") is what gets passed to
        # it, not a direct path to either .onnx file. That directory already
        # existing (vs. insightface's own root=~/.insightface/models/<pack>)
        # is what makes FaceAnalysis.__init__ skip any network download.
        gallery_path = Path(c.get("gallery", "models/face/gallery.json"))
        model_root = Path(c.get("model_root", "models/face"))
        pack_name = str(c.get("pack_name", "ibvap_face"))
        pack_dir = model_root / "models" / pack_name
        det_w = pack_dir / "det_10g.onnx"
        rec_w = pack_dir / "w600k_r50.onnx"
        missing = [p for p in (gallery_path, det_w, rec_w) if not p.exists()]
        if missing:
            logger.warning("Face recognition disabled — missing %s "
                           "(enroll: python training/enroll_faces.py). "
                           "See docs/FACE_RECOGNITION.md", missing[0])
            return

        try:
            gal = json.loads(gallery_path.read_text("utf-8"))
            identities = [
                (g["id"], g.get("name", g["id"]), np.asarray(g["embedding"], dtype=np.float32))
                for g in gal.get("identities", [])
            ]
        except (OSError, ValueError, KeyError) as e:
            logger.error("Face recognition disabled — could not read gallery %s (%s)",
                         gallery_path, e)
            return
        if not identities:
            logger.warning("Face recognition disabled — gallery %s has no identities", gallery_path)
            return
        self._threshold = float(gal.get("match_threshold", self._threshold))

        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            logger.warning("Face recognition disabled — %s (pip install insightface)", e)
            return

        gpu = str(device) not in ("cpu", "-1")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if gpu else ["CPUExecutionProvider"])
        ctx_id = -1 if not gpu else (int(device) if str(device).isdigit() else 0)
        try:
            self._app = FaceAnalysis(
                name=pack_name, root=str(model_root),
                allowed_modules=["detection", "recognition"],
                providers=providers)
            self._app.prepare(ctx_id=ctx_id, det_size=(self._det_size, self._det_size))
        except Exception as e:                                    # pragma: no cover
            logger.error("Face recognition disabled — failed to initialise (%s)", e)
            return

        self._identities = identities
        self.available = True
        logger.info("Face recognition ready — %d identities, threshold %.2f, det_size %d",
                    len(self._identities), self._threshold, self._det_size)

    # ── capture quality ─────────────────────────────────────────────────────
    @staticmethod
    def _clamp01(v: float) -> float:
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))

    def _quality(self, crop: "np.ndarray", det: float, kps) -> float:
        """0..1 score used only to pick the best crop of a burst.

        Sharpness is log-normalised because raw Laplacian variance spans
        roughly 5-2000; linear normalising would let it swamp every other term.
        112 is ArcFace's input side — a face already at least that big gains
        nothing from being bigger.
        """
        import cv2
        h, w = crop.shape[:2]
        if h < 2 or w < 2:
            return 0.0
        area_n = self._clamp01((w * h) ** 0.5 / 112.0)
        small = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        lapvar = float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                                     cv2.CV_64F).var())
        sharp_n = self._clamp01(np.log10(1.0 + lapvar) / np.log10(301.0))
        frontal = 0.5
        if kps is not None and len(kps) >= 2:
            # Landmarks are already computed by detection — eye-to-nose symmetry
            # is a free, decent proxy for yaw.
            (lx, _), (rx, _), (nx, _) = kps[0], kps[1], kps[2] if len(kps) >= 3 else kps[1]
            span = abs(float(rx) - float(lx)) or 1.0
            frontal = 1.0 - self._clamp01(
                abs(abs(float(nx) - float(lx)) - abs(float(rx) - float(nx))) / span)
        wd, wa, ws, wf = self._qw
        return round(self._clamp01(wd * det + wa * area_n + ws * sharp_n + wf * frontal), 3)

    @staticmethod
    def _head_box(x1: int, y1: int, x2: int, y2: int, fw: int, fh: int):
        """Where a head should be inside a YOLO person box, when no face was
        detected. The 0.9*w floor handles seated/crouched people, whose box is
        short and wide and whose head fills a larger fraction of it."""
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        clipped = y1 <= 2
        head_h = 0.30 * h if clipped else max(0.22 * h, 0.9 * w)
        head_w = min(0.62 * w, 1.3 * head_h)
        cx = x1 + w / 2.0
        hx1 = max(0, int(cx - head_w / 2.0))
        hx2 = min(fw, int(cx + head_w / 2.0))
        hy1 = max(0, int(y1))
        hy2 = min(fh, int(y1 + head_h))
        return (hx1, hy1, hx2, hy2), clipped

    def _crop_face(self, frame, box):
        """Pad a face box outward and clip to the frame — a tight detector box
        cuts off chin and hairline, which is what an operator needs to see."""
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * self._face_pad), int((y2 - y1) * self._face_pad)
        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
        cx2, cy2 = min(fw, x2 + px), min(fh, y2 + py)
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            return None
        return frame[cy1:cy2, cx1:cx2].copy()

    def identify_batch(self, per_cam: list) -> dict[int, list[FaceHit]]:
        """per_cam: [(cam_id, frame_bgr, persons[, native_bgr])] where persons
        is a list of (x1, y1, x2, y2, track_id) in **frame** coordinates — the
        CALLER (`Detector._face_pass`) has already filtered this to only the
        tracks due for a capture this tick.

        When `native_bgr` is given (the camera's pre-normalisation frame from
        the same instant) detection runs on it instead, because it has the
        pixels the phone actually captured and face range is decided by pixels
        on the face. Boxes are scaled into it and the returned `bbox` is scaled
        back, so callers always work in frame coordinates — the same convention
        `AnprEngine.submit_batch` uses.

        Returns {cam_id: [FaceHit, ...]} — one entry per due track, always, so
        that a person whose face cannot be detected is still recorded. Entries
        below `det_score_min` carry `trusted=False` and may not vote; entries
        with no detectable face at all carry `source="head_box"`.
        """
        if not self.available:
            return {}
        out: dict[int, list[FaceHit]] = {}
        for item in per_cam:
            cam_id, frame, persons = item[0], item[1], item[2]
            native = item[3] if len(item) > 3 else None
            hits: list[FaceHit] = []
            wh, ww = frame.shape[:2]
            src = native if native is not None else frame
            fh, fw = src.shape[:2]
            sx, sy = fw / max(1, ww), fh / max(1, wh)
            for wx1, wy1, wx2, wy2, tid in persons[: self._max_persons]:
                x1, y1 = int(wx1 * sx), int(wy1 * sy)
                x2, y2 = int(wx2 * sx), int(wy2 * sy)
                bw, bh = x2 - x1, y2 - y1
                px, py = int(bw * self._pad), int(bh * self._pad)
                cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
                cx2, cy2 = min(fw, int(x2 + px)), min(fh, int(y2 + py))
                crop = src[cy1:cy2, cx1:cx2]
                # The person-context crop is taken here too, so all the box
                # geometry — and all the frame-vs-native scaling — stays in one
                # place instead of being repeated by the writer.
                pcrop = (src[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)].copy()
                         if self._save_person_crop and bw > 1 and bh > 1 else None)
                faces = []
                if crop.size and crop.shape[0] >= 20 and crop.shape[1] >= 20:
                    try:
                        faces = self._app.get(crop)
                    except Exception as e:                        # pragma: no cover
                        logger.debug("face detect/embed failed: %s", e)
                        faces = []
                faces = [f for f in faces
                         if float(f.det_score) >= self._capture_score_min]

                if not faces:
                    # Nothing detectable — turned away, too dark, too small,
                    # motion-blurred. Keep a geometric head crop anyway so the
                    # person is never unrecorded; it can never vote.
                    if not (self._always_capture and self._head_box_fallback):
                        continue
                    hbox, clipped = self._head_box(int(x1), int(y1), int(x2), int(y2), fw, fh)
                    hcrop = self._crop_face(src, hbox)
                    if hcrop is None:
                        continue
                    hits.append(FaceHit(
                        person_track=tid, bbox=_unscale(hbox, sx, sy), det_score=0.0,
                        matched_id=None, matched_name=None, similarity=0.0,
                        on_watchlist=False, trusted=False, source="head_box",
                        quality=self._quality(hcrop, 0.15, None), face_crop=hcrop,
                        person_crop=pcrop))
                    continue

                face = max(faces, key=lambda f: float(f.det_score))
                det = float(face.det_score)
                trusted = det >= self._det_score_min
                fx1, fy1, fx2, fy2 = (int(v) for v in face.bbox)
                abs_box = (cx1 + fx1, cy1 + fy1, cx1 + fx2, cy1 + fy2)

                # Only a trusted face is embedded — skipping ArcFace for the
                # low-confidence ones keeps always-capture off the GPU budget,
                # and an untrusted embedding has no one to convince anyway.
                best_id, best_name, best_sim = None, None, 0.0
                if trusted:
                    emb = np.asarray(face.normed_embedding, dtype=np.float32)
                    best_sim = -1.0
                    for gid, gname, gemb in self._identities:
                        sim = float(np.dot(emb, gemb))
                        if sim > best_sim:
                            best_id, best_name, best_sim = gid, gname, sim

                fcrop = self._crop_face(src, abs_box)
                hits.append(FaceHit(
                    person_track=tid, bbox=_unscale(abs_box, sx, sy),
                    det_score=round(det, 3),
                    matched_id=best_id, matched_name=best_name,
                    similarity=round(best_sim, 3),
                    on_watchlist=trusted and best_sim >= self._threshold,
                    trusted=trusted, source="face",
                    quality=self._quality(fcrop, det,
                                          getattr(face, "kps", None)) if fcrop is not None else 0.0,
                    face_crop=fcrop, person_crop=pcrop))
            if hits:
                out[cam_id] = hits
        return out

    def status(self) -> dict:
        return {"enabled": self.available,
                "identities": len(self._identities),
                "threshold": self._threshold}

    def stop(self) -> None:
        pass
