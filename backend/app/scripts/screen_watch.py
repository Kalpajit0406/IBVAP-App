"""
screen_watch.py — DEMO ONLY: point IBVAP at your laptop screen.

Grabs a region of the screen (or the whole primary monitor), runs the same
analysis stack IBVAP uses on camera feeds — YOLO26n detector + yolo26n-pose
skeletons + trained firearm detector (`models/weapon_detector.pt`) + ANPR
number-plate recognition — and paints the results back as a **transparent
click-through overlay** locked over that region. Play a video or open a photo
underneath and IBVAP "sees" what is on screen: a green box + 17-point skeleton on
people, an amber box on vehicles (with a cyan plate readout), a grey box on other
objects, a red box on a held gun (with a `GUN DETECTED` / `ARMED THREAT` banner),
red skeleton when a weapon-ready / lying posture is detected.

This is also reachable from the launch menu: `python server.py` -> [2] Screen ->
on-screen overlay (or `python server.py --mode screen --surface overlay`).

    pip install mss
    python scripts/screen_watch.py                 # whole primary monitor
    python scripts/screen_watch.py --pick          # drag a rectangle first
    python scripts/screen_watch.py --region 200,120,1280,720
    python scripts/screen_watch.py --all           # every COCO class, not the curated set
    python scripts/screen_watch.py --no-pose       # boxes only, no skeletons
    python scripts/screen_watch.py --no-weapon     # skip the firearm detector
    python scripts/screen_watch.py --no-anpr       # skip number-plate recognition
    python scripts/screen_watch.py --monitor 2 --fps 12 --device 0

Esc on the overlay, or Ctrl+C in this terminal, to stop.

Windows only for the transparent overlay (uses Tk `-transparentcolor`). The
detector runs anywhere. This file is self-contained — it does not touch the
server or the pipeline.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path

# Run from anywhere: put the repo root on the path so `import ibvap.*` works,
# and resolve config.yaml relative to the repo, not the current directory.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import yaml

# ── palette (matches the project) ───────────────────────────────────────────
C_PERSON   = "#3cdc3c"
C_VEHICLE  = "#f0aa28"
C_OBJECT   = "#b8b8b8"
C_SKELETON = "#12d7e6"
C_ALERT    = "#ff3b3b"
C_HUD_BG   = "#0c0e12"
C_HUD_FG   = "#e2e8f0"
TRANSPARENT_KEY = "#000001"          # near-black; treated as fully transparent + click-through

# COCO ids
PERSON_ID = 0
VEHICLE_IDS = {1, 2, 3, 4, 5, 6, 7, 8}           # bicycle car motorcycle airplane bus train truck boat
# a curated "what would a border camera care about" set (used unless --all / --classes)
PRESET_CLASSES = sorted(
    {PERSON_ID} | VEHICLE_IDS | {15, 16, 17}          # + cat/dog/horse
    | {24, 25, 26, 28}                                # backpack umbrella handbag suitcase
    | {39, 43, 63, 67, 76}                            # bottle knife laptop cell-phone scissors
)

# COCO 17-keypoint skeleton edges (kept local so this file stands alone)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class Box:
    x1: float; y1: float; x2: float; y2: float
    cls: int; name: str; conf: float; tid: int
    kind: str                       # "person" | "vehicle" | "object" | "weapon"
    kp: object = None               # (17,3) ndarray or None
    posture: str = ""               # e.g. "AIM" / "LYING"
    alert: bool = False
    plate: str = ""                 # ANPR reading on a vehicle box
    plate_conf: float = 0.0


@dataclass
class Snapshot:
    boxes: list = field(default_factory=list)
    grab_ms: float = 0.0
    infer_ms: float = 0.0
    fps: float = 0.0
    n_person: int = 0
    n_vehicle: int = 0
    n_object: int = 0
    stamp: float = 0.0


# ── region picking ─────────────────────────────────────────────────────────
def pick_region() -> tuple[int, int, int, int] | None:
    """Full-screen translucent window; drag a rectangle. Esc → cancel."""
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.25)
    root.attributes("-topmost", True)
    root.configure(bg="#1140ff")
    cv = tk.Canvas(root, cursor="crosshair", bg="#1140ff", highlightthickness=0)
    cv.pack(fill="both", expand=True)
    cv.create_text(root.winfo_screenwidth() // 2, 40, fill="#fff",
                   font=("Segoe UI", 16, "bold"),
                   text="Drag the area IBVAP should watch — Esc to use the whole screen")
    st = {"x0": 0, "y0": 0, "out": None}

    def down(e): st["x0"], st["y0"] = e.x, e.y
    def drag(e):
        cv.delete("r")
        cv.create_rectangle(st["x0"], st["y0"], e.x, e.y, outline="#fff", width=2, tag="r")
    def up(e):
        x1, x2 = sorted((st["x0"], e.x)); y1, y2 = sorted((st["y0"], e.y))
        if x2 - x1 > 30 and y2 - y1 > 30:
            st["out"] = (x1, y1, x2 - x1, y2 - y1)
        root.destroy()

    cv.bind("<Button-1>", down)
    cv.bind("<B1-Motion>", drag)
    cv.bind("<ButtonRelease-1>", up)
    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()
    return st["out"]


# ── capture + detect worker ────────────────────────────────────────────────
class Worker(threading.Thread):
    def __init__(self, region, args, cfg):
        super().__init__(daemon=True)
        self.region = region                       # (x, y, w, h) screen px
        self.args = args
        self.cfg = cfg
        self.snap = Snapshot()
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self._last_dbg = 0.0
        self._wtick = 0                    # frame counter for the weapon pass gate

    def latest(self) -> Snapshot:
        with self._lock:
            return self.snap

    def stop(self):
        self._halt.set()
        if getattr(self, "harv", None) is not None:
            self.harv.stop()
        if getattr(self, "anpr", None) is not None:
            self.anpr.stop()

    # -- setup runs in the worker thread (mss + torch prefer it) --
    def _build(self):
        import mss
        from ultralytics import YOLO

        m = self.cfg.get("model", {})
        self.device = self.args.device or str(m.get("device", "0"))
        self.imgsz = int(self.args.imgsz or m.get("image_size", 640))
        self.conf = float(self.args.conf if self.args.conf is not None else m.get("person_conf", 0.30))
        self.iou = float(m.get("iou", 0.50))
        half = bool(m.get("half", True)) and self.device != "cpu"
        self.quantize = 16 if half else None      # ultralytics predict(): quantize, not half

        weights = self.args.weights or str(_REPO / "yolo26n.pt")
        self.model = YOLO(weights)
        self.classes = None if self.args.all else (
            [int(c) for c in self.args.classes.split(",")] if self.args.classes else PRESET_CLASSES)

        try:
            import supervision as sv
            self.tracker = sv.ByteTrack(frame_rate=max(1, int(self.args.fps)))
            self._sv = sv
        except Exception:
            self.tracker = None
            self._sv = None

        self.pose = None
        self.clf = None
        if not self.args.no_pose:
            try:
                from ibvap.posture import PoseEstimator, PoseClassifier
                p = self.cfg.get("pose", {})
                self.pose = PoseEstimator(
                    weights=p.get("weights", "yolo26n-pose.pt"), device=self.device,
                    half=self.quantize == 16, kp_conf=float(p.get("kp_conf", 0.5)),
                    max_batch=int(p.get("max_batch", 24)), imgsz=int(p.get("imgsz", 256)))
                self.clf = PoseClassifier(self.cfg)
            except Exception as e:
                print(f"[screen_watch] pose disabled ({e})")
                self.pose = None

        import cv2
        self._cv2 = cv2
        self._mss = mss
        self.sct = mss.mss()
        x, y, w, h = self.region
        self._grab_box = {"left": x, "top": y, "width": w, "height": h}
        # detection runs on a width-limited copy; boxes scaled back for the overlay
        self.iw = min(int(self.args.infer_width), w)
        self.scale = w / self.iw
        self.ih = int(round(h / self.scale))

        # warm the exact input shape so the first live frame isn't a ~2 s stall
        blank = np.zeros((self.ih, self.iw, 3), np.uint8)
        for _ in range(2):
            self.model.predict(blank, imgsz=self.imgsz, device=self.device,
                               quantize=self.quantize, classes=self.classes, verbose=False)
        if self.pose is not None:
            self.pose.run_batch([(blank, [(8, 8, 120, 240, 0)])])

        self.harv = None
        if getattr(self.args, "learn", False):
            try:
                from ibvap.learn import Harvester
                self.harv = Harvester(self.cfg)
                print("[screen_watch] --learn: harvesting to data/learning/")
            except Exception as e:
                print(f"[screen_watch] --learn unavailable ({e})")

        # Trained firearm detector — same model the server uses (src/weapon.py).
        self.weapon = None
        if not getattr(self.args, "no_weapon", False):
            try:
                from ibvap.weapon import WeaponDetector
                wc = self.cfg.get("weapon", {}) or {}
                if wc.get("enabled"):
                    w = WeaponDetector(wc, device=self.device, max_batch=1)
                    self.weapon = w if w.available else None
                    print("[screen_watch] weapon detector "
                          + ("on" if self.weapon else "off (weights missing)"))
            except Exception as e:
                print(f"[screen_watch] weapon detector unavailable ({e})")

        # ANPR — same two-stage engine the server uses (src/anpr.py): a plate
        # detector on each vehicle crop + threaded EasyOCR. Needs `pip install
        # easyocr`; self-disables if that or the weights are missing.
        self.anpr = None
        if not getattr(self.args, "no_anpr", False):
            try:
                from ibvap.anpr import AnprEngine
                ac = self.cfg.get("anpr", {}) or {}
                if ac.get("enabled"):
                    a = AnprEngine(ac, device=self.device)
                    self.anpr = a if a.available else None
                    print("[screen_watch] ANPR "
                          + ("on" if self.anpr else "off (easyocr / weights missing)"))
            except Exception as e:
                print(f"[screen_watch] ANPR unavailable ({e})")

    def _grab(self):
        raw = self.sct.grab(self._grab_box)
        frame = np.ascontiguousarray(np.asarray(raw)[:, :, :3])       # BGRA -> BGR
        if self.iw != frame.shape[1]:
            frame = self._cv2.resize(frame, (self.iw, self.ih),
                                     interpolation=self._cv2.INTER_AREA)
        return frame

    def run(self):
        try:
            self._build()
        except Exception as e:
            self.error = e
            self.ready.set()
            return
        self.ready.set()

        sv = self._sv
        period = 1.0 / max(1.0, self.args.fps)
        ema_fps = self.args.fps
        hud_h = int(72 / self.scale) + 1        # our own HUD footprint, in infer px
        hud_w = int(340 / self.scale) + 1
        while not self._halt.is_set():
            t0 = time.perf_counter()
            self._wtick += 1                    # frame counter — gates weapon + ANPR passes
            frame = self._grab()
            frame[:hud_h, :hud_w] = 0           # don't detect our own HUD chip
            t1 = time.perf_counter()

            res = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                     device=self.device, quantize=self.quantize,
                                     classes=self.classes, max_det=300, verbose=False)[0]
            names = res.names
            if sv is not None:
                det = sv.Detections.from_ultralytics(res)
                if self.tracker is not None:
                    det = self.tracker.update_with_detections(det)
                rows = [(det.xyxy[i], int(det.class_id[i]), float(det.confidence[i]),
                         int(det.tracker_id[i]) if det.tracker_id is not None else -1)
                        for i in range(len(det))]
            else:
                b = res.boxes
                rows = [(b.xyxy[i].tolist(), int(b.cls[i]), float(b.conf[i]), -1)
                        for i in range(len(b))]

            boxes: list[Box] = []
            persons, vehicles = [], []
            for xyxy, cls, cf, tid in rows:
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                kind = ("person" if cls == PERSON_ID
                        else "vehicle" if cls in VEHICLE_IDS else "object")
                bx = Box(x1, y1, x2, y2, cls, names.get(cls, str(cls)), cf, tid, kind)
                boxes.append(bx)
                if kind == "person":
                    persons.append((int(x1), int(y1), int(x2), int(y2),
                                    tid if tid >= 0 else len(persons)))
                elif kind == "vehicle" and tid >= 0:
                    vehicles.append((int(x1), int(y1), int(x2), int(y2), tid))

            if self.pose is not None and persons:
                kp_by = self.pose.run_batch([(frame, persons)])
                pmap = {p[4]: b for p, b in zip(persons, [b for b in boxes if b.kind == "person"])}
                for key, kp in kp_by.items():
                    bx = pmap.get(key)
                    if bx is None:
                        continue
                    bx.kp = kp
                    if self.clf is not None:
                        fl = self.clf.classify(key, kp)
                        bx.posture = fl.label
                        bx.alert = fl.any_anomaly
            # weapon pass — infer-res px, same as `persons`; adds red GUN boxes.
            if self.weapon is not None:
                whits = self.weapon.detect_batch([(0, frame, persons)], self._wtick).get(0, [])
                aim_tids = {b.tid for b in boxes
                            if b.kind == "person" and "AIM" in (b.posture or "")}
                for hh in whits:
                    wb = Box(hh.bbox[0], hh.bbox[1], hh.bbox[2], hh.bbox[3],
                             -1, "GUN", hh.conf, hh.person_track, "weapon")
                    wb.alert = bool(hh.confirmed)
                    wb.posture = ("ARMED" if hh.confirmed and hh.person_track in aim_tids
                                  else "GUN" if hh.confirmed else "")
                    boxes.append(wb)

            # ANPR pass — plate detector on vehicle crops + threaded OCR. Reads
            # attach to the vehicle Box; the engine also logs to data/plates/.
            if self.anpr is not None and vehicles:
                self.anpr.submit_batch([(0, frame, vehicles)], self._wtick)
                reads = self.anpr.readings_for_cam(0)
                if reads:
                    live = {tid for *_r, tid in vehicles}
                    for b in boxes:
                        if b.kind == "vehicle" and b.tid in reads:
                            r = reads[b.tid]
                            b.plate, b.plate_conf = r.text, r.conf
                    for tid in [t for t in reads if t not in live]:
                        self.anpr.flush_track((0, tid))
            t2 = time.perf_counter()

            # harvest BEFORE the rescale: boxes/kp are still in `frame` pixel space
            # (object detector only — weapon boxes are class -1, not a YOLO label)
            if self.harv is not None:
                det_boxes = [b for b in boxes if b.kind != "weapon"]
                if det_boxes:
                    self.harv.submit_screen(frame.copy(), det_boxes, self.iw, self.ih,
                                            time.time())

            s = self.scale
            for b in boxes:
                b.x1 *= s; b.y1 *= s; b.x2 *= s; b.y2 *= s
                if b.kp is not None:
                    b.kp = b.kp.copy()
                    b.kp[:, 0] *= s; b.kp[:, 1] *= s

            inst = 1.0 / max(1e-3, time.perf_counter() - t0)
            ema_fps = 0.8 * ema_fps + 0.2 * inst
            snap = Snapshot(
                boxes=boxes, grab_ms=(t1 - t0) * 1000, infer_ms=(t2 - t1) * 1000,
                fps=ema_fps, stamp=time.time(),
                n_person=sum(b.kind == "person" for b in boxes),
                n_vehicle=sum(b.kind == "vehicle" for b in boxes),
                n_object=sum(b.kind == "object" for b in boxes))
            with self._lock:
                self.snap = snap

            if self.args.debug and time.time() - self._last_dbg > 2.0:
                self._last_dbg = time.time()
                top = ", ".join(sorted({b.name for b in boxes})[:6]) or "—"
                print(f"[screen_watch] {ema_fps:4.1f} fps | grab {snap.grab_ms:3.0f} "
                      f"infer {snap.infer_ms:3.0f} ms | P{snap.n_person} V{snap.n_vehicle} "
                      f"O{snap.n_object} | {top}", flush=True)

            dt = time.perf_counter() - t0
            if dt < period:
                self._halt.wait(period - dt)


# ── transparent overlay ────────────────────────────────────────────────────
class Overlay:
    def __init__(self, region, worker: Worker, args):
        self.region = region
        self.worker = worker
        self.args = args
        x, y, w, h = region
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.attributes("-topmost", True)
        try:
            self.root.configure(bg=TRANSPARENT_KEY)
            self.root.attributes("-transparentcolor", TRANSPARENT_KEY)
        except tk.TclError:
            print("[screen_watch] -transparentcolor unsupported here; overlay will be opaque")
        self.cv = tk.Canvas(self.root, width=w, height=h, bg=TRANSPARENT_KEY,
                            highlightthickness=0, bd=0)
        self.cv.pack()
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("q", lambda e: self.quit())
        self._alive = True
        self._deadline = time.time() + args.seconds if args.seconds else None

    def quit(self):
        self._alive = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _draw_hud(self, s: Snapshot):
        cv, W = self.cv, self.region[2]
        cv.create_rectangle(10, 10, 336, 66, fill=C_HUD_BG, outline="#1e2433", width=1)
        cv.create_text(20, 22, anchor="w", fill=C_PERSON, font=("Consolas", 10, "bold"),
                       text="IBVAP · SCREEN WATCH")
        stale = (time.time() - s.stamp) if s.stamp else 9
        tag = "  (starting…)" if not s.stamp else ("  (stalled)" if stale > 1.5 else "")
        cv.create_text(20, 38, anchor="w", fill=C_HUD_FG, font=("Consolas", 9),
                       text=f"{s.fps:4.1f} fps   grab {s.grab_ms:3.0f}ms   infer {s.infer_ms:3.0f}ms{tag}")
        cv.create_text(20, 54, anchor="w", fill=C_HUD_FG, font=("Consolas", 9),
                       text=f"person {s.n_person}   vehicle {s.n_vehicle}   object {s.n_object}")

    def _draw_box(self, b: Box):
        cv = self.cv
        if b.kind == "weapon":
            col = C_ALERT if b.alert else "#c25050"
            wd = 3 if b.alert else 1
            cv.create_rectangle(b.x1, b.y1, b.x2, b.y2, outline=col, width=wd)
            wt = f"{'GUN' if b.alert else 'gun?'} {b.conf:.0%}"
            cv.create_rectangle(b.x1, b.y1 - 15, b.x1 + 7 * len(wt) + 10, b.y1,
                                fill=col, outline="")
            cv.create_text(b.x1 + 5, b.y1 - 8, anchor="w", fill="#0b0b0b",
                           font=("Segoe UI", 8, "bold"), text=wt)
            return
        col = C_ALERT if b.alert else (
            C_PERSON if b.kind == "person" else C_VEHICLE if b.kind == "vehicle" else C_OBJECT)
        cv.create_rectangle(b.x1, b.y1, b.x2, b.y2, outline=col, width=2)
        tag = f"{b.name} {b.conf:.0%}" + (f"  #{b.tid}" if b.tid >= 0 else "")
        tw = 7 * len(tag) + 10
        ty = b.y1 - 15 if b.y1 > 16 else b.y1
        cv.create_rectangle(b.x1, ty, b.x1 + tw, ty + 15, fill=col, outline="")
        cv.create_text(b.x1 + 5, ty + 7, anchor="w", fill="#0b0b0b",
                       font=("Segoe UI", 8, "bold"), text=tag)
        if b.kp is not None:
            self._draw_skeleton(b.kp, C_ALERT if b.alert else C_SKELETON)
        if b.posture:
            cv.create_text(b.x1 + 2, b.y2 + 9, anchor="w", fill=C_ALERT,
                           font=("Segoe UI", 9, "bold"), text=b.posture)
        if b.kind == "vehicle" and b.plate:
            pl = f"{b.plate}  {b.plate_conf:.0%}"
            cv.create_rectangle(b.x1, b.y2 + 2, b.x1 + 8 * len(pl) + 10, b.y2 + 19,
                                fill="#00d7ff", outline="")
            cv.create_text(b.x1 + 5, b.y2 + 11, anchor="w", fill="#0b0b0b",
                           font=("Consolas", 9, "bold"), text=pl)

    def _draw_skeleton(self, kp, col):
        cv = self.cv
        for a, c in SKELETON_EDGES:
            if kp[a, 2] > 0 and kp[c, 2] > 0:
                cv.create_line(kp[a, 0], kp[a, 1], kp[c, 0], kp[c, 1],
                               fill=col, width=2)
        for i in range(17):
            if kp[i, 2] > 0:
                cv.create_oval(kp[i, 0] - 2, kp[i, 1] - 2, kp[i, 0] + 2, kp[i, 1] + 2,
                               fill=col, outline="")

    def tick(self):
        if not self._alive:
            return
        if self._deadline and time.time() > self._deadline:
            return self.quit()
        s = self.worker.latest()
        self.cv.delete("all")
        for b in s.boxes:
            self._draw_box(b)
        self._draw_hud(s)
        self._draw_weapon_banner(s)
        self.root.after(40, self.tick)

    def _draw_weapon_banner(self, s: Snapshot):
        # Overlay coords are screen pixels (Worker already rescaled the boxes).
        conf = [b for b in s.boxes if b.kind == "weapon" and b.alert]
        if not conf:
            return
        text = "ARMED THREAT" if any(b.posture == "ARMED" for b in conf) else "GUN DETECTED"
        cv, W = self.cv, self.region[2]
        cv.create_rectangle(0, 74, W, 110, fill=C_ALERT, outline="")
        cv.create_text(W // 2, 92, fill="#ffffff", font=("Segoe UI", 15, "bold"),
                       text=f"⚠  {text}  -  CRITICAL")

    def run(self):
        try:
            self.root.update_idletasks()
            self.root.focus_force()          # so Esc works before any click-through
        except tk.TclError:
            pass
        self.tick()
        self.root.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser(description="IBVAP screen-share detection overlay (demo)")
    ap.add_argument("--region", help="X,Y,W,H screen pixels (default: whole monitor)")
    ap.add_argument("--pick", action="store_true", help="drag to choose the watch area")
    ap.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary)")
    ap.add_argument("--fps", type=float, default=15.0, help="target detection rate")
    ap.add_argument("--infer-width", type=int, default=1100, help="downscale width for detection")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--weights", default=None, help="default yolo26n.pt")
    ap.add_argument("--device", default=None, help="0 | cpu")
    ap.add_argument("--all", action="store_true", help="detect every COCO class")
    ap.add_argument("--classes", default=None, help="explicit comma id list, e.g. 0,2,7")
    ap.add_argument("--no-pose", action="store_true", help="boxes only, no skeletons")
    ap.add_argument("--no-weapon", action="store_true",
                    help="skip the trained firearm detector (models/weapon_detector.pt)")
    ap.add_argument("--no-anpr", action="store_true",
                    help="skip number-plate recognition (src/anpr.py + easyocr)")
    ap.add_argument("--learn", action="store_true",
                    help="harvest stable, confident detections to data/learning/ (continuous learning)")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds")
    ap.add_argument("--debug", action="store_true", help="print fps / detections to the terminal")
    args = ap.parse_args()

    try:
        cfg = yaml.safe_load(open(_REPO / "config.yaml"))
    except Exception:
        cfg = {}

    if args.pick:
        r = pick_region()
    elif args.region:
        r = tuple(int(v) for v in args.region.split(","))
    else:
        try:
            import mss
            mon = mss.mss().monitors[args.monitor]
        except Exception as e:
            print(f"screen_watch: cannot read monitors ({e}). Try: pip install mss")
            return 1
        r = (mon["left"], mon["top"], mon["width"], mon["height"])
    if not r:
        print("No region chosen."); return 1
    region = tuple(int(v) for v in r)
    print(f"[screen_watch] watching {region[2]}x{region[3]} at ({region[0]},{region[1]}) — "
          f"loading model…  (Esc on the overlay or Ctrl+C here to stop)")

    worker = Worker(region, args, cfg)
    worker.start()
    worker.ready.wait()
    if worker.error is not None:
        print(f"screen_watch: startup failed — {worker.error}")
        return 1

    ov = Overlay(region, worker, args)
    signal.signal(signal.SIGINT, lambda *_: ov.quit())
    try:
        ov.run()
    finally:
        worker.stop()
    print("[screen_watch] stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
