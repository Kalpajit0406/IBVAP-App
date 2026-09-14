"""
calibration.py — per-camera adaptive calibration (live, no training).

While the pipeline runs, this watches what each camera's detector actually does
and gently adjusts two things, per camera, WITHOUT touching model weights:

  * the confidence floor for person / vehicle — raised where confident-but-
    transient false positives keep appearing, lowered (only within `headroom`)
    where real objects sit just under the bar;
  * a suppression mask — coarse grid cells where detections repeatedly appear
    and then die without ever forming a stable track become "known noise" and
    weak detections there are dropped.

It is owned by the inference worker and threaded through model hot-swaps the way
`anpr=` is, so per-camera memory survives a model switch. Pure numpy, no GPU.
Persisted to `data/calibration.json`; reset / freeze from the dashboard.
"""
from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

logger = logging.getLogger("ibvap.calib")

_PERSON = 0
_VEHICLES = {2, 3, 5, 7}
_BINS = 20


def _bin(conf: float) -> int:
    return min(_BINS - 1, max(0, int(float(conf) * _BINS)))


class _Cam:
    __slots__ = ("passes", "gpoint", "maxconf", "cls", "grid_die", "acc", "rej",
                 "areas", "suppress", "person_delta", "vehicle_delta", "n_obs",
                 "frozen", "manual")

    def __init__(self, gw: int, gh: int):
        self.passes: dict[int, int] = {}                 # track_id -> consecutive inferred passes
        self.gpoint: dict[int, tuple] = {}               # track_id -> last ground point (norm)
        self.maxconf: dict[int, float] = {}
        self.cls: dict[int, int] = {}
        self.grid_die = np.zeros((gh, gw), dtype=np.float64)
        self.acc = np.zeros(_BINS, dtype=np.float64)     # conf hist of "became a stable track"
        self.rej = np.zeros(_BINS, dtype=np.float64)     # conf hist of "died young"
        self.areas: dict[int, deque] = {}               # cls -> recent normalised box areas
        self.suppress: set[tuple[int, int]] = set()      # (col, row) grid cells
        self.person_delta = 0.0
        self.vehicle_delta = 0.0
        self.n_obs = 0
        self.frozen = False
        self.manual = False                             # operator set a delta by hand


class Calibrator:
    def __init__(self, config: dict) -> None:
        c = ((config or {}).get("learning", {}) or {}).get("calibration", {}) or {}
        self.enabled = bool(c.get("enabled", True))
        self._alpha = float(c.get("ema_alpha", 0.02))
        self._decay_per_day = float(c.get("decay_per_day", 0.5))
        self._headroom = float(c.get("headroom", 0.05))
        self._max_tighten = float(c.get("max_tighten", 0.30))
        gw, gh = c.get("grid", [32, 18])
        self._gw, self._gh = int(gw), int(gh)
        self._promote = int(c.get("promote_cell_after", 25))
        self._min_obs = int(c.get("min_obs_before_apply", 300))
        self._cams: dict[int, _Cam] = {}
        self._last_decay = time.time()

    def _cam(self, cam_id: int) -> _Cam:
        st = self._cams.get(cam_id)
        if st is None:
            st = _Cam(self._gw, self._gh)
            self._cams[cam_id] = st
        return st

    def _cell(self, gp) -> tuple[int, int]:
        gx, gy = gp
        return (min(self._gw - 1, max(0, int(gx * self._gw))),
                min(self._gh - 1, max(0, int(gy * self._gh))))

    @staticmethod
    def gp_norm(bbox, w: float, h: float) -> tuple[float, float]:
        """Ground point (bottom-centre of the box), normalised 0..1."""
        x1, _y1, x2, y2 = bbox
        return (min(1.0, max(0.0, (x1 + x2) / 2.0 / max(w, 1.0))),
                min(1.0, max(0.0, y2 / max(h, 1.0))))

    @staticmethod
    def area_norm(bbox, w: float, h: float) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, (x2 - x1) * (y2 - y1) / (max(w, 1.0) * max(h, 1.0)))

    # ── observe (worker thread, inferred passes only, ~sub-ms) ────────────
    def observe(self, cam_id: int, detections: list, live_tids: set,
                w: float = 1280.0, h: float = 720.0) -> None:
        if not self.enabled:
            return
        st = self._cam(cam_id)
        live = set(int(t) for t in live_tids)

        for d in detections:
            tid = int(getattr(d, "track_id", -1))
            if tid < 0:
                continue
            st.passes[tid] = st.passes.get(tid, 0) + 1
            st.gpoint[tid] = self.gp_norm(d.bbox, w, h)
            st.maxconf[tid] = max(st.maxconf.get(tid, 0.0), float(d.confidence))
            st.cls[tid] = int(d.class_id)
            st.areas.setdefault(int(d.class_id), deque(maxlen=200)).append(
                self.area_norm(d.bbox, w, h))

        # tracks that vanished this pass → resolve accepted vs died-young
        min_track = 5
        for tid in [t for t in list(st.passes) if t not in live]:
            n = st.passes.pop(tid, 0)
            gp = st.gpoint.pop(tid, None)
            mc = st.maxconf.pop(tid, 0.0)
            st.cls.pop(tid, None)
            st.n_obs += 1
            if n >= min_track:
                st.acc[_bin(mc)] += 1.0
            else:
                st.rej[_bin(mc)] += 1.0
                if gp is not None:
                    col, row = self._cell(gp)
                    st.grid_die[row, col] += 1.0
                    if st.grid_die[row, col] >= self._promote:
                        st.suppress.add((col, row))

        if not st.frozen and not st.manual:
            self._roll_deltas(st)

    def _roll_deltas(self, st: _Cam) -> None:
        base_p, base_v = 0.30, 0.40                       # config person_conf / vehicle_conf
        rej_total = st.rej.sum()
        acc_total = st.acc.sum()
        if rej_total < 5 and acc_total < 5:
            return
        edges = (np.arange(_BINS) + 0.5) / _BINS
        p_rej_above = float(st.rej[edges >= base_p].sum() / max(rej_total, 1.0))
        lo = max(0.0, base_p - self._headroom)
        p_acc_low = float(st.acc[(edges >= lo) & (edges < base_p)].sum() / max(acc_total, 1.0))
        desired = self._max_tighten * p_rej_above - self._headroom * p_acc_low
        desired = float(np.clip(desired, -self._headroom, self._max_tighten))
        st.person_delta = (1 - self._alpha) * st.person_delta + self._alpha * desired
        st.vehicle_delta = (1 - self._alpha) * st.vehicle_delta + self._alpha * desired

    # ── query (worker thread, in Detector._track) ────────────────────────
    def _ready(self, st: _Cam) -> bool:
        return st.manual or st.n_obs >= self._min_obs

    def person_floor(self, cam_id: int, base: float) -> float:
        st = self._cams.get(cam_id)
        if st is None or not self.enabled or not self._ready(st):
            return base
        return float(np.clip(base + st.person_delta, 0.05, 0.95))

    def vehicle_floor(self, cam_id: int, base: float) -> float:
        st = self._cams.get(cam_id)
        if st is None or not self.enabled or not self._ready(st):
            return base
        return float(np.clip(base + st.vehicle_delta, 0.05, 0.95))

    def is_suppressed(self, cam_id: int, gp_norm) -> bool:
        st = self._cams.get(cam_id)
        if st is None or not self.enabled or not st.suppress or not self._ready(st):
            return False
        return self._cell(gp_norm) in st.suppress

    def area_ok(self, cam_id: int, cls: int, area_norm: float) -> bool:
        st = self._cams.get(cam_id)
        if st is None or not self.enabled or not self._ready(st):
            return True
        buf = st.areas.get(int(cls))
        if not buf or len(buf) < 40:
            return True
        arr = np.fromiter(buf, dtype=float)
        lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
        return (lo * 0.2) <= area_norm <= (hi * 5.0)

    # ── housekeeping ─────────────────────────────────────────────────────
    def decay(self) -> None:
        now = time.time()
        days = (now - self._last_decay) / 86400.0
        if days < 1e-3:
            return
        self._last_decay = now
        f = self._decay_per_day ** days
        for st in self._cams.values():
            st.grid_die *= f
            st.acc *= f
            st.rej *= f
            st.suppress = {c for c in st.suppress
                           if st.grid_die[c[1], c[0]] >= self._promote * 0.6}

    def load(self, payload: dict) -> None:
        """From _calib_q. {cam_id?, action: reset|freeze|unfreeze|set, person_delta?, vehicle_delta?}"""
        if not isinstance(payload, dict):
            return
        act = payload.get("action")
        cam = payload.get("cam_id")
        targets = ([self._cam(int(cam))] if cam is not None
                   else list(self._cams.values()))
        for st in targets:
            if act == "reset":
                st.__init__(self._gw, self._gh)
            elif act == "freeze":
                st.frozen = True
            elif act == "unfreeze":
                st.frozen = False; st.manual = False
            elif act == "set":
                if "person_delta" in payload:
                    st.person_delta = float(np.clip(payload["person_delta"],
                                                    -self._headroom, self._max_tighten))
                if "vehicle_delta" in payload:
                    st.vehicle_delta = float(np.clip(payload["vehicle_delta"],
                                                     -self._headroom, self._max_tighten))
                st.manual = True

    @staticmethod
    def validate(payload) -> list[str]:
        if not isinstance(payload, dict):
            return ["body must be an object"]
        errs = []
        if payload.get("action") not in ("reset", "freeze", "unfreeze", "set"):
            errs.append("action: reset|freeze|unfreeze|set")
        for k in ("person_delta", "vehicle_delta"):
            if k in payload and not isinstance(payload[k], (int, float)):
                errs.append(f"{k}: number")
        if "cam_id" in payload and payload["cam_id"] is not None:
            try:
                int(payload["cam_id"])
            except (TypeError, ValueError):
                errs.append("cam_id: int or null")
        return errs

    def snapshot(self) -> dict:
        return {"version": 1, "updated": round(time.time(), 1),
                "cams": {str(cid): {
                    "person_delta": round(st.person_delta, 4),
                    "vehicle_delta": round(st.vehicle_delta, 4),
                    "frozen": st.frozen, "manual": st.manual, "n_obs": st.n_obs,
                    "suppress_cells": sorted(list(st.suppress)),
                    "grid": [self._gw, self._gh],
                } for cid, st in self._cams.items()}}

    def restore(self, payload: dict) -> None:
        """Load a snapshot() dict from disk on startup (deltas + suppression only)."""
        for cid, d in (payload or {}).get("cams", {}).items():
            try:
                st = self._cam(int(cid))
            except (TypeError, ValueError):
                continue
            st.person_delta = float(d.get("person_delta", 0.0))
            st.vehicle_delta = float(d.get("vehicle_delta", 0.0))
            st.frozen = bool(d.get("frozen", False))
            st.manual = bool(d.get("manual", False))
            st.n_obs = int(d.get("n_obs", 0))
            st.suppress = {tuple(c) for c in d.get("suppress_cells", [])}

    def status(self) -> dict:
        return {"enabled": self.enabled, "grid": [self._gw, self._gh],
                "cams": {str(cid): {
                    "person_delta": round(st.person_delta, 3),
                    "vehicle_delta": round(st.vehicle_delta, 3),
                    "suppress_cells": sorted(list(st.suppress)),
                    "frozen": st.frozen, "manual": st.manual, "n_obs": st.n_obs,
                    "applied": self._ready(st),
                } for cid, st in self._cams.items()}}
