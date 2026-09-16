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


@dataclass
class FaceHit:
    person_track: int
    bbox: tuple[int, int, int, int]      # face box, full-frame pixels
    det_score: float
    matched_id: Optional[str]            # gallery identity id, or None
    matched_name: Optional[str]
    similarity: float                    # best cosine similarity vs. the gallery
    on_watchlist: bool                   # similarity >= face.match_threshold


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

    def identify_batch(self, per_cam: list) -> dict[int, list[FaceHit]]:
        """per_cam: [(cam_id, frame_bgr, persons)] where persons is a list of
        (x1, y1, x2, y2, track_id) — the CALLER (`Detector._face_pass`) has
        already filtered this to only the tracks due for a capture this tick.
        Each crop is padded by `face.crop_pad_frac` before detection. Returns
        {cam_id: [FaceHit, ...]} — one entry per crop that had >= 1 face at
        det_score >= det_score_min; a crop with none (turned away, occluded,
        too small) produces no entry for that track."""
        if not self.available:
            return {}
        out: dict[int, list[FaceHit]] = {}
        for cam_id, frame, persons in per_cam:
            hits: list[FaceHit] = []
            fh, fw = frame.shape[:2]
            for x1, y1, x2, y2, tid in persons[: self._max_persons]:
                bw, bh = x2 - x1, y2 - y1
                px, py = int(bw * self._pad), int(bh * self._pad)
                cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
                cx2, cy2 = min(fw, int(x2 + px)), min(fh, int(y2 + py))
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue
                try:
                    faces = self._app.get(crop)
                except Exception as e:                            # pragma: no cover
                    logger.debug("face detect/embed failed: %s", e)
                    continue
                faces = [f for f in faces if float(f.det_score) >= self._det_score_min]
                if not faces:
                    continue
                face = max(faces, key=lambda f: float(f.det_score))
                emb = np.asarray(face.normed_embedding, dtype=np.float32)
                best_id, best_name, best_sim = None, None, -1.0
                for gid, gname, gemb in self._identities:
                    sim = float(np.dot(emb, gemb))
                    if sim > best_sim:
                        best_id, best_name, best_sim = gid, gname, sim
                fx1, fy1, fx2, fy2 = (int(v) for v in face.bbox)
                hits.append(FaceHit(
                    person_track=tid, bbox=(cx1 + fx1, cy1 + fy1, cx1 + fx2, cy1 + fy2),
                    det_score=round(float(face.det_score), 3), matched_id=best_id,
                    matched_name=best_name, similarity=round(best_sim, 3),
                    on_watchlist=best_sim >= self._threshold))
            if hits:
                out[cam_id] = hits
        return out

    def status(self) -> dict:
        return {"enabled": self.available,
                "identities": len(self._identities),
                "threshold": self._threshold}

    def stop(self) -> None:
        pass
