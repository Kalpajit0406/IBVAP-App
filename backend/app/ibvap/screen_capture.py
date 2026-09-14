"""
screen_capture.py — treat this machine's screen as a camera.

`ScreenCapture` is a drop-in sibling of `WebSocketCapture` / `RtspCapture`: it
grabs a region of the screen on its own daemon thread, normalises every frame to
the pipeline size, and exposes the exact same consumer surface the muxer and the
dashboard use (`read()`, `connected`, `socket_open`, `delivered_fps`,
`latency_ms`, `info()`, `stop()`).

This is what powers `python server.py --mode screen`: whatever is on the screen
flows through the *full* pipeline — detector, pose, weapon, risk, geofence,
evidence — and shows on `/monitor` as CAM-00. It is distinct from
`screen_watch.py`, which paints a transparent overlay *on* the screen and never
touches the server.

Needs `mss` (already in requirements.txt). `mss` is created inside the grab
thread — it is not safe to share a handle across threads.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger("ibvap.capture")


class ScreenCapture:
    """Screen grabber with the StreamCapture interface."""

    # Kept in sync with WebSocketCapture.STALE_AFTER — the muxer's batch filter
    # uses that class constant, and `connected` here must agree with it.
    STALE_AFTER = 3.0

    def __init__(self, cam_id: int, region: tuple | None = None, monitor: int = 1,
                 target_fps: float | None = None,
                 norm_w: int = 1280, norm_h: int = 720) -> None:
        self.cam_id = cam_id
        self.label = f"Screen (CAM-{cam_id:02d})"
        self.kind = "screen"
        self._region = tuple(int(v) for v in region) if region else None
        self._monitor = int(monitor)
        self._target_fps = float(target_fps) if target_fps else 24.0
        self._norm_w = int(norm_w)
        self._norm_h = int(norm_h)

        self._queue: queue.Queue = queue.Queue(maxsize=1)   # depth-1, drop-oldest
        self._last_frame_at = 0.0            # time.monotonic() of the last grab
        self._last_arrival = 0.0
        self._fps_ema = 0.0
        self.connected_at = 0.0
        self.frames_received = 0
        self._resized_count = 0
        self._grab_wh = (0, 0)              # actual grabbed pixel size, for info()
        self._err: str | None = None

        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"screen-cap-{cam_id}")

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> "ScreenCapture":
        self.connected_at = time.time()
        self._thread.start()
        logger.info("CAM-%02d screen capture starting (region=%s monitor=%d "
                    "→ %dx%d @ %.0f fps)", self.cam_id,
                    self._region or "full", self._monitor,
                    self._norm_w, self._norm_h, self._target_fps)
        return self

    def stop(self) -> None:
        self._halt.set()

    # ── grab thread ──────────────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            import mss
        except Exception as e:                       # pragma: no cover
            self._err = f"mss not available ({e}); pip install mss"
            logger.error("CAM-%02d screen capture disabled — %s",
                         self.cam_id, self._err)
            return

        period = 1.0 / max(1.0, self._target_fps)
        with mss.mss() as sct:
            if self._region:
                x, y, w, h = self._region
                box = {"left": x, "top": y, "width": w, "height": h}
            else:
                mons = sct.monitors
                idx = self._monitor if 0 <= self._monitor < len(mons) else 1
                box = mons[idx]
            self._grab_wh = (int(box["width"]), int(box["height"]))

            while not self._halt.is_set():
                t0 = time.perf_counter()
                try:
                    raw = sct.grab(box)
                    frame = np.ascontiguousarray(np.asarray(raw)[:, :, :3])  # BGRA→BGR
                except Exception as e:                # pragma: no cover
                    self._err = str(e)
                    self._halt.wait(0.5)
                    continue

                if frame.shape[1] != self._norm_w or frame.shape[0] != self._norm_h:
                    interp = (cv2.INTER_AREA if frame.shape[1] > self._norm_w
                              else cv2.INTER_LINEAR)
                    frame = cv2.resize(frame, (self._norm_w, self._norm_h),
                                       interpolation=interp)
                    self._resized_count += 1

                now = time.monotonic()
                if self._last_arrival:
                    dt = now - self._last_arrival
                    if dt > 0:
                        inst = 1.0 / dt
                        self._fps_ema = (inst if self._fps_ema == 0.0
                                         else 0.9 * self._fps_ema + 0.1 * inst)
                self._last_arrival = now

                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                self._queue.put((now, frame))
                self._last_frame_at = now
                self.frames_received += 1

                dt = time.perf_counter() - t0
                if dt < period:
                    self._halt.wait(period - dt)

    # ── consumer interface ───────────────────────────────────────────────────
    def read(self) -> tuple[float, np.ndarray] | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def connected(self) -> bool:
        return (self._thread.is_alive()
                and (time.monotonic() - self._last_frame_at) < self.STALE_AFTER)

    @property
    def socket_open(self) -> bool:
        # True while the grab thread lives — the muxer never prunes a configured
        # camera whose socket is "open".
        return self._thread.is_alive() and not self._halt.is_set()

    @property
    def delivered_fps(self) -> float:
        if self._last_arrival and (time.monotonic() - self._last_arrival) > 2.0:
            return 0.0
        return round(self._fps_ema, 1)

    @property
    def latency_ms(self) -> float:
        # Screen grab is effectively zero-latency; report a growing gap only if
        # the grab thread has stalled, so the dashboard can flag a freeze.
        if self._last_frame_at and self._thread.is_alive():
            gap = (time.monotonic() - self._last_frame_at) * 1000.0
            return round(gap) if gap > 1500 else 0
        return 0

    def info(self) -> dict:
        gw, gh = self._grab_wh
        return {
            "cam_id": self.cam_id,
            "label": self.label,
            "kind": self.kind,
            "connected": self.connected,
            "socket_open": self.socket_open,
            "connected_at": round(self.connected_at, 1),
            "frames_received": self.frames_received,
            "requested": (f"{gw}x{gh}" if gw else "full screen"),
            "negotiated": (f"{gw}x{gh}@{self._target_fps:.0f}" if gw else "—"),
            "delivered_fps": self.delivered_fps,
            "latency_ms": self.latency_ms,
            "resized": self._resized_count,
            "normalised_to": f"{self._norm_w}x{self._norm_h}",
            "screen": {"region": list(self._region) if self._region else None,
                       "monitor": self._monitor, "error": self._err},
        }
