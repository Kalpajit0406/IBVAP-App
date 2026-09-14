"""
learn.py — continuous-learning data harvester.

While the pipeline runs, every detection that is BOTH high-confidence AND stable
across several detection passes is auto-labelled and written to a candidate pool
(`data/learning/`) as a YOLO-format training example. The operator confirms /
discards / marks-background in the dashboard; `retrain.py` then fine-tunes
`yolo26n.pt` on the kept set.

Nothing here changes model weights. It only writes files:

  data/learning/manifest.jsonl   append-only candidate rows
  data/learning/verdicts.jsonl   append-only operator decisions (last line per id wins)
  data/learning/frames/<id>.jpg  the full training image (raw, normalised)
  data/learning/labels/<id>.txt  "<cls> <cx> <cy> <w> <h>" normalised, one line per box
  data/learning/thumbs/<id>.jpg  ~320px preview for the dashboard
  data/learning/posture.jsonl    keypoints + posture label (feeds `retrain.py --posture`)
  data/learning/plates/          plate crops + plates.jsonl (feeds `retrain.py --anpr`)

Design mirrors the rest of the codebase: a cheap `submit()` on the caller's
thread hands work to one daemon thread that does the JPEG encode + file writes
(like `AnprEngine`'s OCR worker); the manifest append is serialised under a lock
(like `EvidenceChain.append`).
"""
from __future__ import annotations

import json
import logging
import queue
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("ibvap.learn")

_VERDICTS = {"keep", "drop", "background"}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def to_yolo_line(cls: int, bbox_px, w: float, h: float) -> str:
    """(x1,y1,x2,y2) pixels → 'cls cx cy bw bh\\n' normalised 0..1, 6 dp."""
    x1, y1, x2, y2 = (float(v) for v in bbox_px)
    w = max(w, 1.0)
    h = max(h, 1.0)
    cx = min(1.0, max(0.0, (x1 + x2) / 2.0 / w))
    cy = min(1.0, max(0.0, (y1 + y2) / 2.0 / h))
    bw = min(1.0, max(0.0, (x2 - x1) / w))
    bh = min(1.0, max(0.0, (y2 - y1) / h))
    return f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def remap_class(coco_id: int) -> int:
    """Identity today. An explicit seam for a future lean 2-class fine-tune set."""
    return int(coco_id)


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


class Harvester:
    def __init__(self, config: dict) -> None:
        lc = (config or {}).get("learning", {}) or {}
        h = lc.get("harvest", {}) or {}
        self.enabled = bool(lc.get("enabled", True))
        self._min_conf = float(h.get("min_conf", 0.55))
        self._min_track = max(1, int(h.get("min_track_frames", 5)))
        self._ambig = float(h.get("ambiguous_conf", 0.25))
        self._rate_s = float(h.get("per_cam_rate_limit_s", 3.0))
        self._max_pool = int(h.get("max_pool", 5000))
        self._jpeg_q = int(h.get("jpeg_quality", 85))
        self._classes = set(int(c) for c in h.get("classes", [0, 2, 3, 5, 7]))

        root = Path(lc.get("paths", {}).get("pool", "data/learning"))
        self._root = root
        self._frames = root / "frames"
        self._labels = root / "labels"
        self._thumbs = root / "thumbs"
        self._plates = root / "plates"
        self._weapons = root / "weapons"
        for d in (self._frames, self._labels, self._thumbs, self._plates,
                  self._weapons):
            d.mkdir(parents=True, exist_ok=True)
        self._manifest = root / "manifest.jsonl"
        self._verdicts_path = root / "verdicts.jsonl"
        self._posture_path = root / "posture.jsonl"
        self._plates_path = root / "plates.jsonl"
        self._weapon_path = root / "weapon.jsonl"

        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(maxsize=256)
        self._halt = threading.Event()
        self._index: dict[str, dict] = {}          # id -> manifest row (verdict merged in)
        self._verdicts: dict[str, str] = {}
        self._seen: dict[tuple, int] = {}          # (cam_id, track_id) -> consecutive inferred passes
        self._last_grab: dict[object, float] = {}  # cam_id -> last harvest time
        self._last_sig: dict[object, bytes] = {}   # cam_id -> coarse frame signature
        self._screen_prev: list = []               # last screen boxes for the IoU-stability fallback
        self._skipped_ambiguous = 0
        self._harvest_times: list[float] = []
        self._posture_seen: dict[tuple, float] = {}
        self._weapon_seen: dict[tuple, float] = {}

        self._replay()
        self._worker = threading.Thread(target=self._run, daemon=True, name="learn")
        self._worker.start()
        logger.info("Harvester ready — pool=%s  %d existing candidate(s), %d verdict(s)",
                    root, len(self._index), len(self._verdicts))

    # ── load existing pool ─────────────────────────────────────────────────
    def _replay(self) -> None:
        if self._manifest.exists():
            for ln in self._manifest.read_text("utf-8").splitlines():
                try:
                    row = json.loads(ln)
                    self._index[row["id"]] = row
                except (ValueError, KeyError):
                    continue
        self._fold_verdicts()

    def _fold_verdicts(self) -> None:
        """Rebuild self._verdicts from verdicts.jsonl (last line per id wins)."""
        v: dict[str, str] = {}
        if self._verdicts_path.exists():
            for ln in self._verdicts_path.read_text("utf-8").splitlines():
                try:
                    r = json.loads(ln)
                    if r.get("verdict") in _VERDICTS:
                        v[r["id"]] = r["verdict"]
                        if r.get("boxes") and r["id"] in self._index:
                            self._index[r["id"]]["corrected_boxes"] = r["boxes"]
                except (ValueError, KeyError):
                    continue
        self._verdicts = v
        for _id, row in self._index.items():
            row["verdict"] = v.get(_id, "pending")

    # ── stage 1: candidate test (caller's thread — must stay cheap) ────────
    def submit(self, cam_id: int, frame_ref: np.ndarray, detections: list,
               inferred: bool, ts: float, source: str = "cam") -> None:
        if not self.enabled or not inferred or frame_ref is None:
            return

        live = {int(d.track_id) for d in detections if int(getattr(d, "track_id", -1)) >= 0}
        for tid in live:
            self._seen[(cam_id, tid)] = self._seen.get((cam_id, tid), 0) + 1
        for k in [k for k in self._seen if k[0] == cam_id and k[1] not in live]:
            del self._seen[k]

        h, w = frame_ref.shape[:2]
        rows, mrows = [], []
        any_ambiguous = False
        for d in detections:
            cf = float(d.confidence)
            cid = int(d.class_id)
            tid = int(getattr(d, "track_id", -1))
            if self._ambig <= cf < self._min_conf and cid in self._classes:
                any_ambiguous = True
            stable = tid >= 0 and self._seen.get((cam_id, tid), 0) >= self._min_track
            if cf >= self._min_conf and stable and cid in self._classes:
                rows.append(to_yolo_line(remap_class(cid), d.bbox, w, h))
                x1, y1, x2, y2 = d.bbox
                mrows.append({"cls": remap_class(cid), "conf": round(cf, 3),
                              "track_stable": True, "track_id": tid,
                              "xywhn": [round((x1 + x2) / 2 / w, 5), round((y1 + y2) / 2 / h, 5),
                                        round((x2 - x1) / w, 5), round((y2 - y1) / h, 5)]})

        if not rows or any_ambiguous:
            if any_ambiguous:
                self._skipped_ambiguous += 1
            return
        if len(self._pending_ids()) >= self._max_pool:
            return
        now = time.time()
        if now - self._last_grab.get(cam_id, 0.0) < self._rate_s:
            return
        sig = np.ascontiguousarray(frame_ref[::24, ::24, 0]).tobytes()
        if self._last_sig.get(cam_id) == sig:
            return
        self._last_sig[cam_id] = sig
        self._last_grab[cam_id] = now

        _id = "h_" + secrets.token_hex(4)
        meta = {"source": source, "cam_id": int(cam_id), "ts": round(now, 2),
                "wh": [int(w), int(h)], "boxes": mrows,
                "model": getattr(self, "_active_model", "nano")}
        try:
            self._q.put_nowait(("frame", _id, frame_ref.copy(), rows, meta))
        except queue.Full:
            pass

    def submit_screen(self, frame_ref: np.ndarray, boxes: list,
                      iw: int, ih: int, ts: float) -> None:
        """screen_watch path — boxes are screen_watch.Box in infer-res px, no ByteTrack."""
        if not self.enabled or frame_ref is None or not boxes:
            return
        cur = [(b.x1, b.y1, b.x2, b.y2, int(b.cls), float(b.conf)) for b in boxes]
        rows, mrows, any_ambiguous = [], [], False
        for (x1, y1, x2, y2, cid, cf) in cur:
            if self._ambig <= cf < self._min_conf and cid in self._classes:
                any_ambiguous = True
            stable = any(_iou((x1, y1, x2, y2), p[:4]) > 0.6 for p in self._screen_prev)
            if cf >= self._min_conf and stable and cid in self._classes:
                rows.append(to_yolo_line(remap_class(cid), (x1, y1, x2, y2), iw, ih))
                mrows.append({"cls": remap_class(cid), "conf": round(cf, 3),
                              "track_stable": True, "track_id": -1,
                              "xywhn": [round((x1 + x2) / 2 / iw, 5), round((y1 + y2) / 2 / ih, 5),
                                        round((x2 - x1) / iw, 5), round((y2 - y1) / ih, 5)]})
        self._screen_prev = cur
        if not rows or any_ambiguous:
            if any_ambiguous:
                self._skipped_ambiguous += 1
            return
        if len(self._pending_ids()) >= self._max_pool:
            return
        now = time.time()
        if now - self._last_grab.get("screen", 0.0) < self._rate_s:
            return
        self._last_grab["screen"] = now
        _id = "h_" + secrets.token_hex(4)
        meta = {"source": "screen", "cam_id": -1, "ts": round(now, 2),
                "wh": [int(iw), int(ih)], "boxes": mrows, "model": "screen"}
        try:
            self._q.put_nowait(("frame", _id, frame_ref.copy(), rows, meta))
        except queue.Full:
            pass

    def submit_posture(self, cam_id: int, ts: float, box_id: str,
                       kp, label: str) -> None:
        """Feeds `retrain.py --posture`. Rate-limited per track; keeps all anomalies
        and a light sample of normals."""
        if not self.enabled or kp is None:
            return
        key = (cam_id, box_id)
        now = time.time()
        if label == "" and (int(now * 1000) % 100) != 0:      # ~1% of normals
            return
        if now - self._posture_seen.get(key, 0.0) < 2.0:
            return
        self._posture_seen[key] = now
        try:
            arr = np.asarray(kp, dtype=float).reshape(17, 3)
        except (ValueError, TypeError):
            return
        row = {"id": "p_" + secrets.token_hex(4), "cam_id": int(cam_id),
               "ts": round(ts or now, 2), "label": label or "",
               "keypoints": [[round(float(x), 2), round(float(y), 2), round(float(v), 3)]
                             for x, y, v in arr]}
        try:
            self._q.put_nowait(("posture", row))
        except queue.Full:
            pass

    def submit_weapon(self, cam_id: int, frame_ref: np.ndarray, hits: list,
                      ts: float) -> None:
        """Feeds `train_weapon.py`'s operator-review pool. `hits` are
        weapon.WeaponHit; a padded crop of each gun box is saved so the operator
        can confirm / discard it in the dashboard. Rate-limited per person track."""
        if not self.enabled or frame_ref is None or not hits:
            return
        h, w = frame_ref.shape[:2]
        now = time.time()
        for hit in hits:
            key = (cam_id, getattr(hit, "person_track", -1))
            if now - self._weapon_seen.get(key, 0.0) < 2.0:
                continue
            bb = getattr(hit, "bbox", None)
            if not bb:
                continue
            x1, y1, x2, y2 = (int(v) for v in bb)
            px, py = int(0.25 * (x2 - x1)) + 4, int(0.25 * (y2 - y1)) + 4
            cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
            cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            self._weapon_seen[key] = now
            crop = frame_ref[cy1:cy2, cx1:cx2].copy()
            row = {"id": "w_" + secrets.token_hex(4), "cam_id": int(cam_id),
                   "ts": round(ts or now, 2),
                   "conf": round(float(getattr(hit, "conf", 0.0)), 3),
                   "confirmed": bool(getattr(hit, "confirmed", False)),
                   "person_track": int(getattr(hit, "person_track", -1)),
                   "bbox": [x1, y1, x2, y2], "crop_bbox": [cx1, cy1, cx2, cy2]}
            try:
                self._q.put_nowait(("weapon", row, crop))
            except queue.Full:
                pass

    def submit_plate(self, cam_id: int, frame_ref: np.ndarray, reading) -> None:
        """Feeds `retrain.py --anpr`. `reading` is anpr.PlateReading."""
        if not self.enabled or frame_ref is None or reading is None:
            return
        bbox = getattr(reading, "bbox", None)
        if not bbox:
            return
        h, w = frame_ref.shape[:2]
        x1, y1, x2, y2 = (int(max(0, v)) for v in bbox)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 6 or y2 - y1 < 4:
            return
        crop = frame_ref[y1:y2, x1:x2].copy()
        row = {"id": "l_" + secrets.token_hex(4), "cam_id": int(cam_id),
               "ts": round(time.time(), 2),
               "text": getattr(reading, "text", ""), "conf": round(float(getattr(reading, "conf", 0.0)), 3),
               "valid": bool(getattr(reading, "valid", False)),
               "bbox": [x1, y1, x2, y2]}
        try:
            self._q.put_nowait(("plate", row, crop))
        except queue.Full:
            pass

    # ── stage 2: daemon — the only place JPEGs are encoded / files written ──
    def _run(self) -> None:
        while not self._halt.is_set():
            try:
                item = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self._handle(item)
            except Exception as e:                       # never let the daemon die
                logger.debug("harvest write failed: %s", e)
            finally:
                self._q.task_done()

    def _handle(self, item) -> None:
        kind = item[0]
        if kind == "frame":
            _, _id, frame, rows, meta = item
            q = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_q]
            ok, jpg = cv2.imencode(".jpg", frame, q)
            if not ok:
                return
            (self._frames / f"{_id}.jpg").write_bytes(jpg.tobytes())
            (self._labels / f"{_id}.txt").write_text("".join(rows), encoding="utf-8")
            th_w = 320
            th_h = max(1, round(frame.shape[0] * 320 / max(frame.shape[1], 1)))
            ok2, tjpg = cv2.imencode(".jpg", cv2.resize(frame, (th_w, th_h)),
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok2:
                (self._thumbs / f"{_id}.jpg").write_bytes(tjpg.tobytes())
            row = {"id": _id, **meta,
                   "frame": str((self._frames / f"{_id}.jpg").as_posix()),
                   "label": str((self._labels / f"{_id}.txt").as_posix()),
                   "thumb": str((self._thumbs / f"{_id}.jpg").as_posix()),
                   "verdict": "pending"}
            with self._lock:
                with open(self._manifest, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                self._index[_id] = row
                self._harvest_times.append(time.time())
        elif kind == "posture":
            _, row = item
            with self._lock:
                with open(self._posture_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        elif kind == "plate":
            _, row, crop = item
            if getattr(crop, "size", 0):
                cv2.imwrite(str(self._plates / f"{row['id']}.jpg"), crop)
            with self._lock:
                with open(self._plates_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        elif kind == "weapon":
            _, row, crop = item
            if getattr(crop, "size", 0):
                cv2.imwrite(str(self._weapons / f"{row['id']}.jpg"), crop)
            with self._lock:
                with open(self._weapon_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

    # ── review side (fed from POST /api/learn/review via _learn_q) ─────────
    def apply_review(self, id: str, verdict: str, boxes=None) -> None:
        """Re-fold verdicts.jsonl (the durable record written by the route) so a
        dropped queue item is recovered on the next review."""
        with self._lock:
            self._fold_verdicts()
        if id in self._index and verdict in _VERDICTS:
            self._index[id]["verdict"] = verdict
            if boxes:
                self._index[id]["corrected_boxes"] = boxes

    @staticmethod
    def validate_review(payload) -> list[str]:
        if not isinstance(payload, dict):
            return ["body must be an object"]
        errs: list[str] = []
        if not _ID_RE.match(str(payload.get("id", ""))):
            errs.append("id: must match [A-Za-z0-9_-]+")
        if payload.get("verdict") not in _VERDICTS:
            errs.append(f"verdict: one of {sorted(_VERDICTS)}")
        boxes = payload.get("boxes")
        if boxes is not None:
            if not isinstance(boxes, list):
                errs.append("boxes: list or null")
            else:
                for i, b in enumerate(boxes):
                    ok = (isinstance(b, dict) and isinstance(b.get("cls"), int)
                          and isinstance(b.get("xywhn"), list) and len(b["xywhn"]) == 4
                          and all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0
                                  for v in b["xywhn"]))
                    if not ok:
                        errs.append(f"boxes[{i}]: {{cls:int, xywhn:[4 x 0..1]}}")
        return errs

    # ── read side (for /api/learn/pool and retrain.py) ────────────────────
    def _pending_ids(self) -> list[str]:
        return [i for i, r in self._index.items() if r.get("verdict", "pending") == "pending"]

    def pool(self, status: Optional[str] = None) -> dict:
        with self._lock:
            rows = list(self._index.values())
        if status:
            rows = [r for r in rows if r.get("verdict", "pending") == status]
        rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
        counts = {"pending": 0, "keep": 0, "drop": 0, "background": 0}
        for r in self._index.values():
            counts[r.get("verdict", "pending")] = counts.get(r.get("verdict", "pending"), 0) + 1
        return {"items": rows, "counts": counts}

    def kept_items(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._index.values()
                    if r.get("verdict") in ("keep", "background")]

    def thumb_path(self, id: str) -> Optional[Path]:
        if not _ID_RE.match(id):
            return None
        p = self._thumbs / f"{id}.jpg"
        return p if p.exists() else None

    def set_active_model(self, name: str) -> None:
        self._active_model = str(name)

    def status(self) -> dict:
        c = self.pool()["counts"]
        cutoff = time.time() - 60.0
        rate = sum(1 for t in self._harvest_times if t >= cutoff)
        return {"enabled": self.enabled,
                "pool_pending": c.get("pending", 0), "pool_kept": c.get("keep", 0),
                "pool_dropped": c.get("drop", 0), "pool_background": c.get("background", 0),
                "harvested_total": len(self._index), "rate_per_min": rate,
                "max_pool": self._max_pool, "skipped_ambiguous": self._skipped_ambiguous}

    def stop(self) -> None:
        self._halt.set()
