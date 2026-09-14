"""
vehicle_type.py — 8-way vehicle-type classification (Car, Pickup truck, Truck,
Jeep, 2-wheeler, Tanker, Van, Auto-rickshaw) for the event-triggered ANPR
pipeline (ibvap/anpr_events.py).

Self-disabling like ibvap/weapon.py: if models/vehicle_type_classifier.pt
doesn't exist yet (it's produced by training/train_vehicle_type.py, not
shipped), the engine stays "unavailable" and every call falls back to a
coarse mapping from the COCO class the main detector already assigned
(car/motorcycle/bus/truck) — car / two_wheeler / van / truck only. jeep,
pickup_truck, tanker and auto_rickshaw have no COCO analogue and are only ever
reachable once the trained classifier exists.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("ibvap.vehicle_type")

CLASSES = ["car", "pickup_truck", "truck", "jeep", "two_wheeler",
           "tanker", "van", "auto_rickshaw"]

# COCO id -> best-effort coarse label, used only while no trained model exists
# (or a crop's top prediction is below min_conf). 2=car, 3=motorcycle, 5=bus,
# 7=truck — matches ibvap.detector._VEHICLES.
_COCO_FALLBACK = {2: "car", 3: "two_wheeler", 5: "van", 7: "truck"}


@dataclass
class VehicleTypeResult:
    label: str
    conf: float           # 0.0 for a pure fallback
    fallback: bool
    ts: float = 0.0


class VehicleTypeClassifier:
    def __init__(self, cfg: dict, device: str = "0") -> None:
        self.available = False
        cfg = cfg or {}
        self._fallback_on = bool(cfg.get("fallback_coco", True))
        self._min_conf = float(cfg.get("min_conf", 0.35))
        self._imgsz = int(cfg.get("imgsz", 224))
        self._device = device
        self._model = None
        self._names: dict[int, str] = {}    # model index -> label, from the checkpoint
        self._classified = 0
        self._fallbacks = 0

        weights = cfg.get("weights", "models/vehicle_type_classifier.pt")
        if not Path(weights).exists():
            logger.warning(
                "Vehicle-type classifier disabled — weights not found at %s "
                "(coarse COCO fallback stays active until training/"
                "train_vehicle_type.py produces one)", weights)
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(weights))
        except Exception as e:
            logger.error("Vehicle-type classifier disabled — failed to load %s: %s",
                        weights, e)
            return

        # Index->label comes from the CHECKPOINT, never from CLASSES' order.
        # Ultralytics builds a classification model's class list by sorting the
        # ImageFolder directory names alphabetically, which is not the order
        # CLASSES is written in — indexing CLASSES with the model's top1 would
        # mislabel every prediction (a car would come back "pickup_truck").
        names = getattr(self._model, "names", None) or {}
        self._names = {int(i): str(n) for i, n in dict(names).items()}
        if not self._names:
            logger.error("Vehicle-type classifier disabled — %s has no class "
                         "names embedded", weights)
            self._model = None
            return
        unknown = sorted(set(self._names.values()) - set(CLASSES))
        if unknown:
            logger.warning("Vehicle-type checkpoint has classes outside the "
                           "taxonomy: %s — they'll be reported verbatim", unknown)
        missing = sorted(set(CLASSES) - set(self._names.values()))
        if missing:
            logger.warning("Vehicle-type checkpoint cannot predict %s — it was "
                           "trained without them; those stay unreachable", missing)

        self.available = True
        logger.info("Vehicle-type classifier ready — %s, %d classes (%s), device=%s",
                    Path(weights).name, len(self._names),
                    ", ".join(self._names[i] for i in sorted(self._names)), device)

    def _fallback(self, coco_class_id: int, now: float) -> VehicleTypeResult:
        self._fallbacks += 1
        if not self._fallback_on:
            return VehicleTypeResult("unknown", 0.0, True, now)
        return VehicleTypeResult(_COCO_FALLBACK.get(coco_class_id, "unknown"),
                                 0.0, True, now)

    def classify_batch(self, items: list[tuple[int, int, np.ndarray, int]]
                       ) -> dict[tuple[int, int], VehicleTypeResult]:
        """items: (cam_id, track_id, crop_bgr, coco_class_id) — one crop per
        vehicle arrival. One predict() call across the whole batch, mirroring
        AnprEngine.submit_batch's one-forward-pass-per-tick shape."""
        now = time.time()
        out: dict[tuple[int, int], VehicleTypeResult] = {}
        if not items:
            return out

        if not self.available:
            for cam_id, tid, _crop, coco_cid in items:
                out[(int(cam_id), int(tid))] = self._fallback(coco_cid, now)
            return out

        crops = [c for *_, c, _ in items]
        try:
            preds = self._model.predict(crops, imgsz=self._imgsz,
                                        device=self._device, verbose=False)
        except Exception as e:
            logger.debug("vehicle-type predict failed, using fallback: %s", e)
            for cam_id, tid, _crop, coco_cid in items:
                out[(int(cam_id), int(tid))] = self._fallback(coco_cid, now)
            return out

        for (cam_id, tid, _crop, coco_cid), pred in zip(items, preds):
            key = (int(cam_id), int(tid))
            probs = getattr(pred, "probs", None)
            if probs is None:
                out[key] = self._fallback(coco_cid, now)
                continue
            top1 = int(probs.top1)
            conf = float(probs.top1conf)
            label = self._names.get(top1)
            if conf < self._min_conf or label is None:
                out[key] = self._fallback(coco_cid, now)
                continue
            self._classified += 1
            out[key] = VehicleTypeResult(label, conf, False, now)
        return out

    def flush_track(self, key: tuple[int, int]) -> None:
        pass   # no per-track state kept here beyond a single classify_batch call

    def status(self) -> dict:
        # Report what the loaded checkpoint can actually predict, not the
        # aspirational taxonomy — they diverge whenever a model was trained on
        # a subset (see the class-coverage warnings logged at load).
        classes = ([self._names[i] for i in sorted(self._names)]
                   if self.available else CLASSES)
        return {"enabled": self.available, "classes": classes,
                "taxonomy": CLASSES,
                "classified": self._classified, "fallbacks": self._fallbacks}
