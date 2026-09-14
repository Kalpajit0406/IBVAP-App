"""
IBVAP Web Server — camera intake (phones + RTSP) + live monitor dashboard.

Run:
    python server.py

    Monitor        → http://<LAN-ip>:8090/monitor
    Phone (LAN)    → https://<LAN-ip>:8443/camera/0   (HTTPS required by getUserMedia)
    Phone (remote) → python scripts/tunnel.py, then open the /camera/0 URL it prints

Each stream ID (0, 1, 2 …) is a separate camera. Phone slots and pulled RTSP
cameras are declared in config.yaml `streams:` and share one batched pipeline.
The dashboard polls /status; there is no server-push channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import secrets
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse,
                               Response, StreamingResponse)

from ibvap.calibration import Calibrator
from ibvap.detector import Detector
from ibvap.event_store import EventStore
from ibvap.evidence import EvidenceChain
from ibvap.geofence import GeoFenceEngine
from ibvap.learn import Harvester
from ibvap.risk_engine import RiskEngine
from ibvap.rtsp_capture import RtspCapture, redact
from ibvap.screen_capture import ScreenCapture
from ibvap.security import (TOKEN_HEADER, ListenerGuard, is_authorized,
                            load_or_create_token)
from ibvap.snapshots import SnapshotWriter, malicious_postures
from ibvap.ws_capture import WebSocketCapture

# Windows consoles default to cp1252 and mangle non-ASCII log output
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ibvap.server")

# ── Shared state (module-level so detection thread can reach it) ───────────────
captures: dict[int, WebSocketCapture] = {}


def _native_frame_fn(cam_id: int, ts: float | None = None):
    """(ts, native-resolution frame) for one camera, or None if it hasn't
    delivered a frame yet — or its capture kind doesn't expose one (e.g.
    ScreenCapture). Passed into Detector so a vehicle-arrival snapshot
    (ibvap/anpr_events.py) can use the camera's true resolution instead of the
    1280x720 working frame every other pass operates on. Called only once per
    vehicle arrival, never per tick."""
    cap = captures.get(cam_id)
    if cap is None:
        return None
    getter = getattr(cap, "get_native_frame", None)
    if getter is None:
        return None
    try:
        return getter(ts)
    except TypeError:                       # capture without timestamp lookup
        return getter()


_detector = None                       # set by the detection thread once built
_event_store = None                    # set by the inference worker; closed on shutdown
_harvester = None                      # continuous-learning data harvester
_calibrator = None                     # per-camera adaptive calibration
_snapshots = None                      # intrusion / malicious-movement snapshot writer
cfg: dict = {}

# Input source chosen at launch (python server.py [--mode ...]). "all" keeps the
# historical behaviour: open every config.yaml `streams:` entry.
#   screen → grab this machine's screen as CAM-00 (ibvap/screen_capture.py)
#   phone  → open only the phone (ws) slots
#   cctv   → open only the pulled RTSP/webcam/file streams
#   all    → everything in config.yaml
_STARTUP_MODE = "all"
_SCREEN_REGION: tuple | None = None    # (x, y, w, h) for --mode screen, else full monitor
_SCREEN_MONITOR = 1

# Ingest normalisation target — phones that don't honour the requested
# resolution are rescaled to this before entering the pipeline.
NORM_W, NORM_H = 1280, 720

# Latest raw annotated frame per camera (ndarray) — used to build the mosaic
_latest_bgr: dict[int, "np.ndarray"] = {}
# Latest metadata per camera
_camera_meta: dict[int, dict] = {}
# Virtual-fence overlay state, written by the worker, read by /status and the
# mosaic builder. Whole-entry replace — lock-free, same as _camera_meta.
_geofence_fences: dict[int, list] = {}     # cam_id -> [{id,kind,points,label,direction}]
_geofence_active: dict[int, dict] = {}     # cam_id -> {fence_id: is_breaching}

# Hand-off from the muxer thread to the inference worker thread. Depth 1: the
# worker always gets the freshest snapshot; if it falls behind, the muxer drops
# the stale one rather than letting a backlog grow. This is what keeps the
# blocking GPU predict() (≈28 ms for the TensorRT engine) off the muxer, so the
# muxer holds its 24 Hz tick and never backpressures the camera sockets.
_infer_q: "queue.Queue[list]" = queue.Queue(maxsize=1)
_switch_q: "queue.Queue" = queue.Queue(maxsize=1)     # str profile name | dict {weights,...}
# New fence set from POST /api/fences, handed to the worker to reload live.
_fences_q: "queue.Queue[list]" = queue.Queue(maxsize=1)
# Continuous learning: operator review verdicts, and calibration commands.
_learn_q: "queue.Queue[dict]" = queue.Queue(maxsize=64)
_calib_q: "queue.Queue[dict]" = queue.Queue(maxsize=8)

STATIC = Path("static")



# ── Lifecycle ─────────────────────────────────────────────────────────────────
_detection_started = False


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Both the HTTP and HTTPS listeners share this app, so the lifespan fires
    # twice — the detection thread must only ever start once.
    global cfg, _detection_started, NORM_W, NORM_H
    if not _detection_started:
        _detection_started = True
        cfg = yaml.safe_load(open("config.yaml"))
        ing = cfg.get("ingest", {}) or {}
        NORM_W = int(ing.get("normalise_width", NORM_W))
        NORM_H = int(ing.get("normalise_height", NORM_H))
        _load_streams_overlay()
        _open_configured_streams()
        threading.Thread(target=_inference_worker, daemon=True, name="infer").start()
        threading.Thread(target=_muxer_loop, daemon=True, name="muxer").start()
        logger.info("IBVAP ready — muxer + inference worker started "
                    "(ingest normalised to %dx%d)", NORM_W, NORM_H)
    try:
        yield
    finally:
        for cap in list(captures.values()):
            try:
                cap.stop()
            except Exception:
                pass
        if _detector is not None and getattr(_detector, "anpr", None) is not None:
            try:
                _detector.anpr.stop()
            except Exception:
                pass
        if _detector is not None and getattr(_detector, "weapon", None) is not None:
            try:
                _detector.weapon.stop()
            except Exception:
                pass
        if _harvester is not None:
            try:
                _harvester.stop()
            except Exception:
                pass
        if _calibrator is not None:
            try:
                _persist_calibration(_calibrator)
            except Exception:
                pass
        if _event_store is not None:
            try:
                _event_store.close()
            except Exception:
                pass


# Values in a stream's `url` that mean "a phone will connect here" rather than
# "pull this source". Empty / missing also means a phone slot.
_WS_URLS = {"", "ws", "phone", "mobile", "browser"}


# Dynamic camera edits from the dashboard/app (Add/Edit/Delete Camera) are
# persisted here, NOT to config.yaml — mirrors data/fences.json. config.yaml
# stays the pristine, hand-commented seed; once data/streams.json exists it is
# the source of truth for `cfg["streams"]`, exactly like fences.
def _streams_path() -> Path:
    return Path("data/streams.json")


def _load_streams_overlay() -> None:
    p = _streams_path()
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text("utf-8"))
        streams = data.get("streams")
        if isinstance(streams, list):
            cfg["streams"] = streams
            logger.info("Loaded %d stream(s) from %s", len(streams), p)
    except (ValueError, OSError) as e:
        logger.warning("Could not read %s: %s", p, e)


def _save_streams() -> None:
    p = _streams_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"streams": cfg.get("streams", [])}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, p)


def _open_configured_streams() -> None:
    """Turn config.yaml `streams:` into live captures, filtered by the launch mode.

    A stream with a real `url` (rtsp://, http://, a webcam index, a file) is
    pulled by an RtspCapture on its own decode thread. A stream with no url —
    or url: ws — is a WebSocketCapture slot a phone can connect to. `--mode
    screen` ignores config.yaml entirely and grabs this machine's screen as
    CAM-00. All of them land in the same `captures` dict and the same batched
    pipeline.
    """
    mode = _STARTUP_MODE

    # ── screen: one synthetic camera, no config.yaml streams ────────────────
    if mode == "screen":
        target_fps = float(cfg.get("model", {}).get("stream_fps", 24.0))
        captures[0] = ScreenCapture(0, region=_SCREEN_REGION,
                                    monitor=_SCREEN_MONITOR, target_fps=target_fps,
                                    norm_w=NORM_W, norm_h=NORM_H).start()
        # Make RiskEngine / the muxer's prune set treat CAM-00 as a real,
        # configured camera (zone_sensitivity feeds the risk score).
        cfg.setdefault("streams", [])
        if not any(int(s.get("id", -1)) == 0 for s in cfg["streams"]):
            cfg["streams"].append({"id": 0, "name": "Screen", "url": "screen",
                                   "zone_sensitivity": 0.6})
        logger.info("Input mode: SCREEN — grabbing the screen as CAM-00")
        return

    c = cfg.get("cctv", {}) or {}
    opened_ws = opened_pull = 0
    for s in cfg.get("streams", []):
        cam_id = int(s["id"])
        if not s.get("enabled", True):
            logger.info("CAM-%02d disabled in config — skipping", cam_id)
            continue
        url = str(s.get("url", "")).strip()
        is_ws = url.lower() in _WS_URLS
        if url.lower() == "screen":
            target_fps = float(cfg.get("model", {}).get("stream_fps", 24.0))
            captures[cam_id] = ScreenCapture(
                cam_id, region=_SCREEN_REGION, monitor=_SCREEN_MONITOR,
                target_fps=target_fps, norm_w=NORM_W, norm_h=NORM_H).start()
            opened_pull += 1
            continue
        if mode == "phone" and not is_ws:
            continue
        if mode == "cctv" and is_ws:
            continue
        if is_ws:
            captures[cam_id] = WebSocketCapture(cam_id, norm_w=NORM_W, norm_h=NORM_H)
            opened_ws += 1
            continue
        try:
            captures[cam_id] = RtspCapture(
                cam_id, url,
                name=s.get("name"),
                norm_w=NORM_W, norm_h=NORM_H,
                transport=str(s.get("transport", c.get("transport", "tcp"))),
                decode_fps=float(s.get("decode_fps", c.get("decode_fps", 15))),
                reconnect_delay=float(c.get("reconnect_delay", 3.0)),
                stall_timeout=float(c.get("stall_timeout", 8.0)),
                open_timeout=float(c.get("open_timeout", 8.0)),
            ).start()
            opened_pull += 1
        except Exception as e:
            logger.error("CAM-%02d failed to start (%s): %s",
                         cam_id, redact(url), e)

    if mode == "cctv" and opened_pull == 0:
        logger.warning("Input mode: REAL CAMERAS — but config.yaml has no "
                       "rtsp:// / webcam / file streams. Fill in `streams:` "
                       "(see docs/CCTV_INTEGRATION.md). Falling back to phone slots.")
        for s in cfg.get("streams", []):
            cam_id = int(s["id"])
            if s.get("enabled", True) and str(s.get("url", "")).strip().lower() in _WS_URLS:
                if captures.setdefault(cam_id, WebSocketCapture(
                        cam_id, norm_w=NORM_W, norm_h=NORM_H)):
                    opened_ws += 1
    logger.info("Input mode: %s — %d phone slot(s), %d pulled stream(s)",
                mode.upper(), opened_ws, opened_pull)


app = FastAPI(title="IBVAP", lifespan=_lifespan)

_security: dict | None = None     # resolved on first request — cfg loads in _lifespan


def _security_policy() -> dict:
    global _security
    if _security is None:
        sc = cfg.get("security", {}) or {}
        _security = {
            "token": load_or_create_token(sc.get("token_file", "data/api_token")),
            "allow_loopback": bool(sc.get("allow_loopback_writes", True)),
            "require_token": bool(sc.get("require_token", True)),
        }
    return _security


@app.middleware("http")
async def _write_guard(request, call_next):
    pol = _security_policy()
    client = request.client.host if request.client else None
    if not is_authorized(request.method, request.url.path, client,
                         request.headers.get(TOKEN_HEADER), pol["token"],
                         allow_loopback=pol["allow_loopback"],
                         require_token=pol["require_token"]):
        logger.warning("Refused %s %s from %s — missing/invalid API token",
                       request.method, request.url.path, client)
        return JSONResponse(
            {"error": "write access denied — send the API token in the "
                      "X-IBVAP-Token header (see data/api_token on the server)"},
            status_code=401)
    return await call_next(request)


# ── Camera intake ─────────────────────────────────────────────────────────────
# Both /cam/N and /camera/N serve the phone page. The page always opens the
# WebSocket at /ws/camera/N (cam_id is injected server-side), so the short path
# works behind a custom domain without any client change.
@app.get("/cam/{cam_id}", response_class=HTMLResponse)
@app.get("/camera/{cam_id}", response_class=HTMLResponse)
async def camera_page(cam_id: int) -> HTMLResponse:
    html = (STATIC / "camera.html").read_text()
    return HTMLResponse(html.replace("{{CAM_ID}}", str(cam_id)))


# Slots the operator removed while the server runs. A phone page reconnects
# every 2 s on its own, so without this a removed phone camera would come
# straight back. Re-adding the stream (POST /api/streams) clears the entry.
_removed_cams: set[int] = set()
_ws_conns: dict[int, WebSocket] = {}


@app.websocket("/ws/camera/{cam_id}")
async def camera_ws(ws: WebSocket, cam_id: int) -> None:
    await ws.accept()
    if cam_id in _removed_cams:
        await ws.close(code=4404, reason="camera slot removed by the operator")
        return
    cap = captures.setdefault(
        cam_id, WebSocketCapture(cam_id, norm_w=NORM_W, norm_h=NORM_H))
    generation = cap.open()
    _ws_conns[cam_id] = ws
    logger.info("CAM-%02d connected", cam_id)
    try:
        while True:
            # A message is either a JSON text "hello" (device metadata) or a
            # binary JPEG frame. The browser sends exactly one hello, first.
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                cap.push_frame(data)
            elif (text := msg.get("text")) is not None:
                try:
                    info = json.loads(text)
                    if info.get("type") == "hello":
                        server_ms = cap.set_hello(info)
                        if server_ms is not None:
                            # Echo the server clock so the phone can align its
                            # own timestamps and show a live latency figure.
                            await ws.send_text(json.dumps({
                                "type": "synced",
                                "serverTime": server_ms,
                                "t0": info.get("t0"),
                            }))
                except (ValueError, TypeError):
                    logger.debug("CAM-%02d: non-JSON text frame ignored", cam_id)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("CAM-%02d socket error: %s", cam_id, e)
    finally:
        # Only the current socket may mark the camera down; a reconnect that
        # raced ahead of this teardown keeps its own live state.
        if _ws_conns.get(cam_id) is ws:
            _ws_conns.pop(cam_id, None)
        if cap.close(generation):
            logger.info("CAM-%02d disconnected (%d frames received)",
                        cam_id, cap.frames_received)


# ── Monitor ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return RedirectResponse("/monitor")


@app.get("/monitor", response_class=HTMLResponse)
async def monitor_page() -> HTMLResponse:
    return HTMLResponse(
        (STATIC / "monitor.html").read_text(),
        headers={"Cache-Control": "no-store"},
    )


# ── MJPEG mosaic — every camera in one grid, one connection ──────────────────
# The grid is composed here, in the request's own worker thread (run_in_executor),
# NOT on the detection thread. Compositing 8x 720p frames + JPEG encoding costs
# ~15-25 ms; doing it on the detection loop stole enough time to drop the muxer
# below real-time. The detector only keeps _latest_bgr fresh.
@app.get("/stream")
async def mjpeg_mosaic():
    logger.info("MJPEG mosaic requested")
    loop = asyncio.get_running_loop()

    async def generate():
        try:
            while True:
                if _latest_bgr:
                    jpeg = await loop.run_in_executor(None, _build_mosaic)
                    if jpeg:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n"
                               b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                               + jpeg + b"\r\n")
                await asyncio.sleep(0.045)    # ~22 fps mosaic (fine for 2-4 cams)
        except asyncio.CancelledError:
            logger.info("MJPEG mosaic closed")
            raise

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache", "Connection": "close"},
    )


# ── MJPEG stream — one camera, for debugging (dashboard uses the mosaic) ──────
@app.get("/stream/{cam_id}")
async def mjpeg_stream(cam_id: int):
    logger.info("MJPEG stream requested for CAM-%02d", cam_id)
    loop = asyncio.get_running_loop()

    def _encode(cid: int):
        f = _latest_bgr.get(cid)
        if f is None:
            return None
        ok, jpeg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes() if ok else None

    async def generate():
        try:
            while True:
                jpeg = await loop.run_in_executor(None, _encode, cam_id)
                if jpeg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                           + jpeg + b"\r\n")
                await asyncio.sleep(0.045)   # ~22 fps single-cam stream
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache",
                 "Connection": "close"},
    )


@app.get("/meta/{cam_id}")
async def cam_meta(cam_id: int):
    return JSONResponse(_camera_meta.get(cam_id, {}))


# ── Mosaic layout (operator focus / grid) ────────────────────────────────────
# The mosaic is one shared server-composited JPEG, so its layout is global:
# whoever sets it sets it for everyone watching /stream. The next /stream frame
# and the next /status poll pick it up — no reconnect.
@app.post("/api/layout", status_code=202)
async def set_layout(payload: dict = Body(...)):
    mode = str(payload.get("mode", "grid")).lower()
    if mode not in ("grid", "focus"):
        return JSONResponse({"error": "mode: grid|focus"}, status_code=400)
    mains = payload.get("mains") or []
    if not isinstance(mains, list) or not all(isinstance(v, int) for v in mains):
        return JSONResponse({"error": "mains: list[int]"}, status_code=400)
    _stats["layout"] = {"mode": mode, "mains": mains[:4]}
    logger.info("Mosaic layout: %s%s", mode,
                f" mains={mains[:4]}" if mode == "focus" else "")
    return {"layout": _stats["layout"]}


# ── Detection loop (background thread) ────────────────────────────────────────
# Shared counters for /status endpoint
_stats: dict = {"frames": 0}


def _device_rows() -> list[dict]:
    """Per-device view for the dashboard: identity, link state, live/idle, res/fps."""
    gate = _detector.gate_stats() if _detector is not None else {}
    rows = []
    for cid, cap in sorted(captures.items()):
        g = gate.get(cid, {})
        base = cap.info()
        # "live" here means motion gating is currently letting frames reach
        # YOLO. A connected-but-static camera is "idle" — no GPU spent on it.
        base["state"] = ("idle" if base["connected"] and not g.get("live", False)
                         else "live" if base["connected"] else "offline")
        base["infer_count"] = g.get("infer_count", 0)
        base["gated_count"] = g.get("gated_count", 0)
        base["gate_skip_pct"] = g.get("skip_pct", 0.0)
        base["last_motion"] = g.get("last_motion", 0.0)
        rows.append(base)
    return rows


@app.get("/devices")
async def devices():
    return JSONResponse({"count": len(captures), "devices": _device_rows()})


@app.post("/api/reconnect/{cam_id}", status_code=202)
async def reconnect_camera(cam_id: int):
    """Force an RTSP camera to drop and reopen — for a frozen feed mid-demo."""
    cap = captures.get(cam_id)
    if cap is None or not hasattr(cap, "request_reconnect"):
        return JSONResponse(
            {"error": f"CAM-{cam_id:02d} is not a pullable camera"},
            status_code=400)
    cap.request_reconnect()
    logger.info("CAM-%02d reconnect requested via API", cam_id)
    return {"reconnecting": cam_id}


# ── Streams & Camera Intake Management ────────────────────────────────────────
# Dynamic add/edit/delete persists to data/streams.json (see _save_streams /
# _load_streams_overlay above) — config.yaml is never rewritten.

def _open_single_stream(s: dict) -> None:
    cam_id = int(s["id"])
    url = str(s.get("url", "")).strip()
    is_ws = url.lower() in _WS_URLS
    existing = captures.get(cam_id)

    # A phone may already be actively streaming into this slot. Its lifecycle
    # is owned by the /ws/camera/{id} handler's own WebSocketCapture reference;
    # replacing captures[cam_id] here would orphan that live socket's queue
    # (frames keep arriving into the old object, nothing reads them from the
    # new one). An edit that only changes e.g. the label must not drop it.
    if is_ws and isinstance(existing, WebSocketCapture) and existing.connected:
        logger.info("CAM-%02d is a live phone stream — metadata updated, "
                    "socket left untouched", cam_id)
        return

    if existing is not None:
        try:
            existing.stop()
        except Exception:
            pass
        captures.pop(cam_id, None)

    if not s.get("enabled", True):
        logger.info("CAM-%02d is disabled", cam_id)
        return

    c = cfg.get("cctv", {}) or {}

    if url.lower() == "screen":
        target_fps = float(cfg.get("model", {}).get("stream_fps", 24.0))
        captures[cam_id] = ScreenCapture(
            cam_id, region=_SCREEN_REGION, monitor=_SCREEN_MONITOR,
            target_fps=target_fps, norm_w=NORM_W, norm_h=NORM_H).start()
    elif is_ws:
        captures[cam_id] = WebSocketCapture(cam_id, norm_w=NORM_W, norm_h=NORM_H)
    else:
        try:
            captures[cam_id] = RtspCapture(
                cam_id, url,
                name=s.get("name"),
                norm_w=NORM_W, norm_h=NORM_H,
                transport=str(s.get("transport", c.get("transport", "tcp"))),
                decode_fps=float(s.get("decode_fps", c.get("decode_fps", 15))),
                reconnect_delay=float(c.get("reconnect_delay", 3.0)),
                stall_timeout=float(c.get("stall_timeout", 8.0)),
                open_timeout=float(c.get("open_timeout", 8.0)),
            ).start()
        except Exception as e:
            logger.error("CAM-%02d failed to start (%s): %s", cam_id, redact(url), e)


@app.get("/api/streams")
async def list_streams():
    ip = _lan_ip()
    configured = cfg.get("streams", [])
    rows = []
    for s in configured:
        cid = int(s["id"])
        cap = captures.get(cid)
        url = str(s.get("url", "")).strip()
        is_ws = url.lower() in _WS_URLS
        stype = "phone" if is_ws else "screen" if url.lower() == "screen" else "rtsp" if url.startswith("rtsp://") else "http" if url.startswith("http://") else "webcam" if url.isdigit() else "file"
        rows.append({
            "id": cid,
            "name": s.get("name", f"CAM-{cid:02d}"),
            "url": redact(url) if url and not is_ws else url,
            "raw_url": url,
            "type": stype,
            "enabled": s.get("enabled", True),
            "zone_sensitivity": s.get("zone_sensitivity", 0.5),
            "transport": s.get("transport", "tcp"),
            "decode_fps": s.get("decode_fps", 15),
            "live": cap.connected if cap is not None else False,
            "frames_received": cap.frames_received if cap is not None else 0,
            "phone_url": f"https://{ip}:{HTTPS_PORT}/cam/{cid}" if is_ws else None,
        })
    return JSONResponse({
        "lan_ip": ip,
        "https_port": HTTPS_PORT,
        "lan_phone_intake": bool(_stats.get("lan_phone_intake", False)),
        "http_port": HTTP_PORT,
        "streams": rows,
    })


@app.post("/api/streams", status_code=202)
async def upsert_stream(payload: dict = Body(...)):
    """Add or edit a camera stream and hot-load it into the pipeline."""
    if "id" not in payload:
        # Assign next available ID
        existing_ids = {int(s["id"]) for s in cfg.get("streams", [])}
        cam_id = 0
        while cam_id in existing_ids:
            cam_id += 1
        payload["id"] = cam_id
    else:
        cam_id = int(payload["id"])

    name = str(payload.get("name", f"CAM-{cam_id:02d}")).strip()
    url = str(payload.get("url", "ws")).strip()
    zone_sens = float(payload.get("zone_sensitivity", 0.6))
    transport = str(payload.get("transport", "tcp")).lower()
    decode_fps = float(payload.get("decode_fps", 15.0))
    enabled = bool(payload.get("enabled", True))

    entry = {
        "id": cam_id,
        "name": name,
        "url": url,
        "zone_sensitivity": zone_sens,
        "transport": transport,
        "decode_fps": decode_fps,
        "enabled": enabled,
    }

    streams = cfg.setdefault("streams", [])
    idx = next((i for i, s in enumerate(streams) if int(s.get("id", -1)) == cam_id), None)
    if idx is not None:
        streams[idx] = entry
    else:
        streams.append(entry)
    streams.sort(key=lambda s: int(s.get("id", 0)))
    _removed_cams.discard(cam_id)

    # RtspCapture.stop() blocks up to ~3s joining its decode thread — keep
    # that off the event loop so /status and other requests don't stall.
    await asyncio.get_running_loop().run_in_executor(
        None, _open_single_stream, entry)
    _save_streams()
    logger.info("Stream saved and loaded: CAM-%02d (%s)", cam_id, redact(url))
    return {"saved": entry}


@app.delete("/api/streams/{cam_id}", status_code=202)
async def delete_stream(cam_id: int):
    """Remove a camera: drop it from configuration, close its capture, hang up
    a connected phone and take its tile off the mosaic."""
    streams = cfg.get("streams", [])
    was_configured = any(int(s.get("id", -1)) == cam_id for s in streams)
    if not was_configured and cam_id not in captures and cam_id not in _latest_bgr:
        return JSONResponse({"error": f"CAM-{cam_id:02d} does not exist"}, status_code=404)
    cfg["streams"] = [s for s in streams if int(s.get("id", -1)) != cam_id]
    _removed_cams.add(cam_id)
    ws = _ws_conns.pop(cam_id, None)
    if ws is not None:
        try:
            await ws.close(code=4404, reason="camera slot removed by the operator")
        except Exception:
            pass
    cap = captures.pop(cam_id, None)
    if cap is not None:
        def _safe_stop():
            try:
                cap.stop()
            except Exception:
                pass
        await asyncio.get_running_loop().run_in_executor(None, _safe_stop)
    _latest_bgr.pop(cam_id, None)
    _camera_meta.pop(cam_id, None)
    _geofence_active.pop(cam_id, None)
    _geofence_fences.pop(cam_id, None)
    lay = _stats.get("layout") or {}
    if lay.get("mains"):
        lay["mains"] = [c for c in lay["mains"] if c != cam_id]
    if was_configured:
        _save_streams()
    logger.info("Stream removed: CAM-%02d%s", cam_id,
                "" if was_configured else " (unconfigured device)")
    return {"deleted": cam_id, "configured": was_configured}


@app.post("/api/streams/probe")
async def probe_stream(payload: dict = Body(...)):
    """Test whether an RTSP, HTTP, webcam index or video file URL is reachable."""
    url = str(payload.get("url", "")).strip()
    transport = str(payload.get("transport", "tcp")).lower()
    if not url:
        return JSONResponse({"ok": False, "error": "URL is required"}, status_code=400)

    if url.lower() in _WS_URLS or url.lower() == "screen":
        return JSONResponse({
            "ok": True,
            "note": "Phone / screen slots aren't network streams — nothing to probe. "
                    "Save the camera and it will appear once connected.",
        })

    loop = asyncio.get_running_loop()

    def _test():
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}|stimeout;3000000|max_delay;500000")
        src = int(url) if url.isdigit() else url
        api = cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(src, api)
        open_time = time.perf_counter() - t0
        if not cap.isOpened():
            return {
                "ok": False,
                "error": f"Failed to connect after {open_time:.1f}s. Check IP, credentials, port 554, or transport.",
            }
        ok, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if not ok or frame is None:
            return {
                "ok": False,
                "error": "Opened connection but failed to read video frame.",
            }
        return {
            "ok": True,
            "width": w if w > 0 else frame.shape[1],
            "height": h if h > 0 else frame.shape[0],
            "fps": fps if fps > 0 else 24.0,
            "latency_ms": round(open_time * 1000, 1),
        }

    try:
        # OpenCV's FFmpeg stimeout option is unreliable on an unroutable/
        # firewalled host — bound the HTTP response ourselves so "Test" never
        # spins indefinitely. The executor thread may still be blocked behind
        # the scenes until the OS-level TCP timeout fires; that's a one-off
        # leaked probe, not a hang the operator has to wait through.
        res = await asyncio.wait_for(loop.run_in_executor(None, _test), timeout=6.0)
    except asyncio.TimeoutError:
        return JSONResponse({
            "ok": False,
            "error": "Timed out after 6s — host unreachable, wrong port, or "
                     "blocked by a firewall/VLAN.",
        })
    return JSONResponse(res)


@app.post("/api/streams/discover")
async def discover_lan_cameras():
    """Discover ONVIF cameras and NVRs on the local network via WS-Discovery."""
    loop = asyncio.get_running_loop()

    def _run_wsd():
        try:
            from tools.discover_cameras import discover
            return discover(timeout=3.0)
        except Exception as e:
            logger.warning("ONVIF discovery error: %s", e)
            return []

    devices = await loop.run_in_executor(None, _run_wsd)
    return JSONResponse({"count": len(devices), "devices": devices})


@app.get("/api/mobile-info")
async def mobile_info():
    ip = _lan_ip()
    have_cert = Path("cert.pem").exists() and Path("key.pem").exists()
    slots = []
    for s in cfg.get("streams", []):
        cid = int(s["id"])
        url = str(s.get("url", "")).strip()
        if url.lower() in _WS_URLS:
            cap = captures.get(cid)
            slots.append({
                "id": cid,
                "name": s.get("name", f"Phone-{cid}"),
                "url": f"https://{ip}:{HTTPS_PORT}/cam/{cid}",
                "connected": cap.connected if cap is not None else False,
                "frames_received": cap.frames_received if cap is not None else 0,
            })
    return JSONResponse({
        "lan_ip": ip,
        "https_port": HTTPS_PORT,
        "lan_phone_intake": bool(_stats.get("lan_phone_intake", False)),
        "http_port": HTTP_PORT,
        "have_cert": have_cert,
        "slots": slots,
    })


def _registry_path() -> Path:
    return Path(cfg.get("learning", {}).get("paths", {})
               .get("registry", "models/registry.json"))


def _read_registry() -> list:
    p = _registry_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text("utf-8"))
    except (ValueError, OSError):
        return []


def _push_switch(item) -> None:
    try:
        _switch_q.get_nowait()
    except queue.Empty:
        pass
    try:
        _switch_q.put_nowait(item)
    except queue.Full:
        pass
    _stats["switching"] = True


def _weapon_registry_path() -> Path:
    return Path(cfg.get("weapon", {}).get("registry", "models/weapon_registry.json"))


def _read_weapon_registry() -> list:
    p = _weapon_registry_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text("utf-8"))
    except (ValueError, OSError):
        return []


def _scan_model_files() -> list[dict]:
    """Scan models/ and root for all available weights (.pt, .engine, .onnx)."""
    found = []
    seen = set()
    for root_dir in (Path("models"), Path(".")):
        if not root_dir.exists():
            continue
        for ext in ("*.pt", "*.engine", "*.onnx"):
            for p in root_dir.glob(ext):
                rel = p.as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                st = p.stat()
                found.append({
                    "name": p.stem,
                    "file": rel,
                    "filename": p.name,
                    "size_mb": round(st.st_size / (1024 * 1024), 2),
                    "modified": st.st_mtime,
                    "type": p.suffix.lstrip("."),
                })
    return sorted(found, key=lambda x: x["name"])


@app.post("/api/switch-model", status_code=202)
async def switch_model(payload: dict = Body(...)):
    # 1. A fine-tuned model from the continuous-learning / thermal registry
    ref = str(payload.get("weights_ref", "")).strip()
    if ref:
        entry = next((e for e in _read_registry() if e.get("name") == ref), None)
        if entry is None:
            return JSONResponse({"error": f"Unknown weights_ref '{ref}'"}, status_code=400)
        _push_switch({"weights": entry["weights"],
                      "pose_weights": entry.get("pose_weights", "yolo26n-pose.pt"),
                      "label": entry.get("label", ref), "name": ref})
        logger.info("Model switch requested: fine-tuned '%s'", ref)
        return {"queued": ref}

    # 2. Direct weights file (for any newly trained or custom model file)
    weights = str(payload.get("weights", "")).strip()
    if weights:
        pose_weights = str(payload.get("pose_weights", "yolo26n-pose.pt")).strip()
        label = str(payload.get("label", Path(weights).stem)).strip()
        name = str(payload.get("name", Path(weights).stem)).strip()
        _push_switch({"weights": weights,
                      "pose_weights": pose_weights,
                      "label": label, "name": name})
        logger.info("Model switch requested: custom weights '%s'", weights)
        return {"queued": name}

    # 3. A built-in model_profiles entry
    profile = str(payload.get("profile", "")).strip().lower()
    profiles = cfg.get("model_profiles", {})
    if profile not in profiles:
        return JSONResponse(
            {"error": f"Unknown profile '{profile}'. "
                      f"Valid profiles: {list(profiles.keys())}"},
            status_code=400)
    _push_switch(profile)
    _stats["switch_target"] = profile
    logger.info("Model switch requested: %s", profile)
    return {"queued": profile}


@app.get("/api/models")
async def list_models():
    profiles = {k: {"label": v.get("label", k)}
                for k, v in cfg.get("model_profiles", {}).items()}
    registry_list = _read_registry()
    for e in registry_list:
        profiles[e["name"]] = {
            "label": e.get("label", e["name"]), "ref": True,
            "delta": e.get("delta"), "base_mAP": (e.get("base_mAP") or {}).get("mAP50_95"),
            "ft_mAP": (e.get("ft_mAP") or {}).get("mAP50_95"),
            "kind": e.get("kind", "fine-tuned"),
            "dataset": e.get("dataset", ""),
            "created": e.get("created"),
            "weights": e.get("weights"),
        }
    weapon_list = _read_weapon_registry()
    model_files = _scan_model_files()

    return JSONResponse({
        "active": _stats.get("active_model", "nano"),
        "switching": _stats.get("switching", False),
        "profiles": profiles,
        "registry": registry_list,
        "weapons": weapon_list,
        "weapon_active": (_detector.weapon.status()
                          if _detector is not None and getattr(_detector, "weapon", None)
                          else {"enabled": False}),
        "anpr_active": (_detector.anpr.status()
                        if _detector is not None and getattr(_detector, "anpr", None)
                        else {"enabled": False}),
        "files": model_files,
    })


@app.post("/api/shutdown")
async def shutdown_server():
    logger.info("Shutdown requested via API")
    loop = asyncio.get_running_loop()
    loop.call_later(0.5, lambda: os._exit(0))
    return {"status": "shutting_down"}


# ── Virtual fences ───────────────────────────────────────────────────────────
# The operator draws polygon zones / tripwire lines on each camera view in the
# dashboard; they are persisted to config.yaml `geofence.store` and handed to
# the inference worker over `_fences_q` (the same producer/consumer shape as
# the model hot-swap). Geometry is normalised 0..1 image coords per cam_id.
def _fences_path() -> Path:
    return Path(cfg.get("geofence", {}).get("store", "data/fences.json"))


def _read_fences() -> list:
    p = _fences_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text("utf-8")).get("fences", [])
    except (ValueError, OSError) as e:
        logger.warning("Could not read fences from %s: %s", p, e)
        return []


def _write_fences(fences: list) -> None:
    p = _fences_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"fences": fences}, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _queue_fences(fences: list) -> None:
    try:
        _fences_q.get_nowait()
    except queue.Empty:
        pass
    try:
        _fences_q.put_nowait(fences)
    except queue.Full:
        pass


@app.get("/api/fences")
async def get_fences(cam_id: int | None = None):
    fences = _read_fences()
    if cam_id is not None:
        fences = [f for f in fences if int(f.get("cam_id", -1)) == cam_id]
    return JSONResponse({"fences": fences})


@app.post("/api/fences", status_code=202)
async def put_fences(payload: dict = Body(...)):
    """Full-set replace: the dashboard holds the whole list and re-POSTs it."""
    fences = payload.get("fences")
    if not isinstance(fences, list):
        return JSONResponse({"errors": ["body must be {\"fences\": [...]}"]},
                            status_code=400)
    errs = GeoFenceEngine.validate(fences)
    if errs:
        return JSONResponse({"errors": errs}, status_code=400)
    now = time.time()
    for f in fences:
        f.setdefault("id", "f_" + secrets.token_hex(3))
        f.setdefault("created_at", now)
    _write_fences(fences)
    _queue_fences(fences)
    logger.info("Fences saved: %d fence(s)", len(fences))
    return {"saved": len(fences)}


@app.delete("/api/fences/{fence_id}", status_code=202)
async def delete_fence(fence_id: str):
    fences = [f for f in _read_fences() if f.get("id") != fence_id]
    _write_fences(fences)
    _queue_fences(fences)
    logger.info("Fence deleted: %s (%d remain)", fence_id, len(fences))
    return {"deleted": fence_id}


# ── Intrusion snapshots ─────────────────────────────────────────────────────
# JPEGs written by the inference worker (ibvap/snapshots.py) on a fence breach,
# a confirmed weapon, or a malicious posture, one folder per camera under
# snapshots.dir. Read-only here — the dashboard ALERT LOG shows the thumbnails.
_SNAP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.jpg$")


def _snapshots_dir() -> Path:
    return Path(cfg.get("snapshots", {}).get("dir", "data/snapshots"))


@app.get("/snap/{cam_id}/{name}")
async def snapshot_file(cam_id: int, name: str):
    if not _SNAP_NAME.match(name):
        return Response(status_code=404)
    base = (_snapshots_dir() / str(cam_id)).resolve()
    p = (base / name).resolve()
    try:
        p.relative_to(base)
    except ValueError:
        return Response(status_code=404)
    if not p.is_file():
        return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/snapshots")
async def list_snapshots(cam_id: int | None = None, limit: int = 30):
    limit = max(1, min(int(limit), 200))
    root = _snapshots_dir()
    if cam_id is not None:
        cams = [cam_id]
    else:
        cams = sorted(int(p.name) for p in root.glob("*")
                      if p.is_dir() and p.name.isdigit()) if root.exists() else []
    out: list = []
    for c in cams:
        idx = root / str(c) / "index.jsonl"
        if not idx.exists():
            continue
        try:
            lines = idx.read_text("utf-8").splitlines()[-limit:]
        except OSError:
            continue
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    out.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return JSONResponse({"snapshots": out[:limit]})


# ── Incident history + evidence integrity ───────────────────────────────────
# The hash chain and the SQLite incident log were write-only from the operator's
# point of view — verification needed a shell on the server (main.py
# --verify-chain). These expose both read-only. Verification opens its own
# reader (never EvidenceChain.__init__, which may repair a crash fragment) and
# runs off the event loop because it hashes the whole file.
def _chain_path() -> Path:
    return Path(cfg.get("evidence", {}).get("hash_chain_path", "data/hash_chain.jsonl"))


def _chain_reader() -> EvidenceChain:
    r = EvidenceChain.__new__(EvidenceChain)
    r._path = _chain_path()
    return r


@app.get("/api/evidence/verify")
async def evidence_verify():
    t0 = time.time()
    s = await asyncio.to_thread(_chain_reader().summary)
    s["checked_at"] = round(time.time(), 3)
    s["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    if not s["ok"]:
        logger.error("EVIDENCE CHAIN INTEGRITY FAILURE at line %s", s["bad_line"])
    return JSONResponse(s)


@app.get("/api/evidence/recent")
async def evidence_recent(limit: int = 50):
    limit = max(1, min(int(limit), 500))
    rows = await asyncio.to_thread(_chain_reader().tail, limit)
    return JSONResponse({"records": rows})


@app.get("/api/events")
async def list_events(limit: int = 100, cam_id: int | None = None,
                      level: str | None = None, since: float | None = None):
    if _event_store is None:
        return JSONResponse({"events": [], "counts": {}, "available": False})
    rows = await asyncio.to_thread(_event_store.query, limit, cam_id, level, since)
    counts = await asyncio.to_thread(_event_store.counts_by_level, since)
    return JSONResponse({"events": rows, "counts": counts, "available": True})


# ── Event-triggered ANPR results ────────────────────────────────────────────
# One picture + one JSON record per vehicle arrival (plate + vehicle type +
# both confidences), written by ibvap/anpr_events.py to data/anpr_results/
# <cam_id>/ — deliberately separate from data/snapshots/ (breach/weapon/
# posture). Read-only here; the dashboard/app show the thumbnails.
def _anpr_results_dir() -> Path:
    return Path(cfg.get("anpr_results", {}).get("dir", "data/anpr_results"))


@app.get("/anpr_snap/{cam_id}/{name}")
async def anpr_snap_file(cam_id: int, name: str):
    if not _SNAP_NAME.match(name):
        return Response(status_code=404)
    base = (_anpr_results_dir() / str(cam_id)).resolve()
    p = (base / name).resolve()
    try:
        p.relative_to(base)
    except ValueError:
        return Response(status_code=404)
    if not p.is_file():
        return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/anpr/results")
async def list_anpr_results(cam_id: int | None = None, limit: int = 30):
    limit = max(1, min(int(limit), 200))
    root = _anpr_results_dir()
    if cam_id is not None:
        cams = [cam_id]
    else:
        cams = sorted(int(p.name) for p in root.glob("*")
                      if p.is_dir() and p.name.isdigit()) if root.exists() else []
    out: list = []
    for c in cams:
        idx = root / str(c) / "index.jsonl"
        if not idx.exists():
            continue
        try:
            lines = idx.read_text("utf-8").splitlines()[-limit:]
        except OSError:
            continue
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    out.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return JSONResponse({"results": out[:limit]})


# ── Continuous learning ─────────────────────────────────────────────────────
_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_LEARN_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _calibration_path() -> Path:
    return Path(cfg.get("learning", {}).get("paths", {})
               .get("calibration", "data/calibration.json"))


def _persist_calibration(cal) -> None:
    p = _calibration_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cal.snapshot(), indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _restore_calibration(cal) -> None:
    p = _calibration_path()
    if p.exists():
        try:
            cal.restore(json.loads(p.read_text("utf-8")))
            logger.info("Calibration restored from %s", p)
        except (ValueError, OSError) as e:
            logger.warning("Could not restore calibration: %s", e)


@app.get("/api/learn/pool")
async def learn_pool(status: str | None = None):
    if _harvester is None:
        return JSONResponse({"items": [], "counts": {}})
    return JSONResponse(_harvester.pool(status))


@app.get("/learn/thumb/{item_id}.jpg")
async def learn_thumb(item_id: str):
    if _harvester is None or not _LEARN_ID.match(item_id):
        return Response(status_code=404)
    p = _harvester.thumb_path(item_id)
    if p is None:
        return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/learn/review", status_code=202)
async def learn_review(payload: dict = Body(...)):
    if _harvester is None:
        return JSONResponse({"error": "learning disabled"}, status_code=400)
    errs = Harvester.validate_review(payload)
    if errs:
        return JSONResponse({"errors": errs}, status_code=400)
    row = {"id": payload["id"], "verdict": payload["verdict"],
           "boxes": payload.get("boxes"), "ts": time.time()}
    p = Path(cfg.get("learning", {}).get("paths", {}).get("pool", "data/learning")) / "verdicts.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    try:
        _learn_q.put_nowait(row)
    except queue.Full:
        pass
    return {"reviewed": payload["id"], "verdict": payload["verdict"]}


@app.post("/api/learn/calibration", status_code=202)
async def learn_calibration(payload: dict = Body(...)):
    if _calibrator is None:
        return JSONResponse({"error": "calibration disabled"}, status_code=400)
    errs = Calibrator.validate(payload)
    if errs:
        return JSONResponse({"errors": errs}, status_code=400)
    try:
        _calib_q.put_nowait(payload)
    except queue.Full:
        pass
    logger.info("Calibration command: %s", payload.get("action"))
    return {"queued": payload.get("action")}


@app.post("/api/learn/retrain", status_code=202)
async def learn_retrain(payload: dict = Body(...)):
    mode = str(payload.get("mode", "detector"))
    if mode not in ("detector", "posture", "anpr", "thermal", "weapon"):
        return JSONResponse({"error": "mode: detector|posture|anpr|thermal|weapon"}, status_code=400)
    if _stats.get("last_train", {}).get("state") == "running":
        return JSONResponse({"error": "a training run is already in progress"},
                            status_code=409)

    script_map = {
        "detector": "training/retrain.py",
        "posture": "training/retrain.py",
        "anpr": "training/train_anpr.py",
        "thermal": "training/train_thermal.py",
        "weapon": "training/train_weapon.py",
    }
    script = script_map.get(mode, "training/retrain.py")
    dry_run = bool(payload.get("dry_run"))

    # Each script has its own CLI surface — forwarding a flag a script doesn't
    # recognise makes argparse exit(2) immediately (silent, near-instant
    # "training" that never trained anything). Build only what each accepts.
    flags: list[str] = []
    if script == "training/retrain.py":
        # continuous-learning fine-tune: --dry-run / --posture / --anpr only —
        # epochs/batch/freeze for THIS script live in config.yaml learning.retrain.
        if mode == "posture":
            flags.append("--posture")
        if dry_run:
            flags.append("--dry-run")
    else:
        # train_anpr.py / train_thermal.py / train_weapon.py share --dry-run,
        # --epochs, --batch (+ --src/--imgsz, left at their config defaults).
        if mode == "anpr":
            target = str(payload.get("target", "detector")).strip().lower()
            flags.append("--ocr" if target == "ocr" else "--detector")
        if dry_run:
            flags.append("--dry-run")
        if payload.get("epochs") is not None:
            flags.extend(["--epochs", str(int(payload["epochs"]))])
        if payload.get("batch") is not None:
            flags.extend(["--batch", str(int(payload["batch"]))])
        if mode == "thermal" and payload.get("freeze") is not None:
            # only train_thermal.py accepts --freeze (train_weapon.py / train_anpr.py don't)
            flags.extend(["--freeze", str(int(payload["freeze"]))])

    _stats["last_train"] = {"state": "running", "mode": mode,
                            "script": script,
                            "started": time.time(), "log_tail": []}
    threading.Thread(target=_run_retrain, args=(script, flags), daemon=True,
                     name="retrain").start()
    return {"started": mode, "script": script, "flags": flags}


def _run_retrain(script: str, flags: list) -> None:
    import subprocess
    lt = _stats["last_train"]
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        proc = subprocess.Popen([sys.executable, "-u", script, *flags],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                cwd=str(Path(__file__).parent), env=child_env)
        ring: list[str] = []
        for line in proc.stdout:                       # type: ignore[union-attr]
            line = line.rstrip()
            ring.append(line)
            del ring[:-40]
            lt["log_tail"] = list(ring)
            for prefix in ("RETRAIN_RESULT ", "THERMAL_RESULT ", "WEAPON_RESULT ", "ANPR_RESULT "):
                if line.startswith(prefix):
                    try:
                        lt.update(json.loads(line[len(prefix):]))
                    except ValueError:
                        pass
        rc = proc.wait()
        lt["state"] = "done" if rc == 0 else "failed"
        lt["finished"] = time.time()
    except Exception as e:                             # pragma: no cover
        lt["state"] = "failed"
        lt["error"] = str(e)
        lt["finished"] = time.time()


@app.get("/status")
async def status():
    return JSONResponse({
        "device": _stats.get("device", "unknown"),
        "active_model": _stats.get("active_model", "nano"),
        "switching": _stats.get("switching", False),
        "cameras": {cid: {"live": cap.connected,
                          "socket_open": cap.socket_open,
                          "frames_received": cap.frames_received}
                    for cid, cap in captures.items()},
        "devices": _device_rows(),
        "frames_processed": _stats["frames"],
        "inference_ms": round(_stats.get("infer_ms", 0.0), 1),
        "fps": round(_stats.get("fps", 0.0), 1),
        "pipeline": {
            "muxer_tps": round(_stats.get("mux_tps", 0.0), 1),
            "worker_bps": round(_stats.get("worker_bps", 0.0), 1),
            "snapshots_dropped": _stats.get("dropped", 0),
            "mean_batch": round(_stats.get("mean_batch", 0.0), 2),
            "last_batch": _stats.get("last_batch", 0),
            "detector_passes": _stats.get("detector_passes", 0),
            "frames_inferred": _stats.get("frames_inferred", 0),
            "frames_gated": _stats.get("frames_gated", 0),
            "frames_rate_skipped": _stats.get("frames_rate_skipped", 0),
            "gpu_saving_pct": round(_stats.get("gpu_saving_pct", 0.0), 1),
        },
        "buffered_bgr": {cid: True for cid in _latest_bgr},
        "meta": _camera_meta,
        "anpr": (_detector.anpr.status()
                 if _detector is not None and getattr(_detector, "anpr", None)
                 else {"enabled": False}),
        "vehicle_type": (_detector.vehicle_type.status()
                         if _detector is not None and getattr(_detector, "vehicle_type", None)
                         else {"enabled": False}),
        "anpr_results": (_detector.anpr_results.status()
                         if _detector is not None and getattr(_detector, "anpr_results", None)
                         else {"enabled": False}),
        "weapon": (_detector.weapon.status()
                   if _detector is not None and getattr(_detector, "weapon", None)
                   else {"enabled": False}),
        "input_mode": _STARTUP_MODE,
        "geofence": {
            "enabled": bool(cfg.get("geofence", {}).get("enabled")),
            "fences": _geofence_fences,     # {cam_id: [{id,kind,points,label,direction}]}
            "active": _geofence_active,     # {cam_id: {fence_id: is_breaching}}
        },
        "learning": ({**_harvester.status(),
                      "calibration": (_calibrator.status() if _calibrator else {"enabled": False}),
                      "last_train": _stats.get("last_train", {})}
                     if _harvester is not None else {"enabled": False}),
        "snapshots": (_snapshots.status() if _snapshots is not None
                      else {"enabled": False}),
        # The dashboard maps a click on the mosaic back to a camera + pixel; it
        # must use the exact rect map the server composed, never re-derive it.
        "mosaic": _mosaic_status(),
    })


def _mosaic_status() -> dict:
    ids = sorted(_latest_bgr)
    cw, ch, tiles = _mosaic_tiles(ids)
    lay = _stats.get("layout", {}) or {}
    return {
        "ids": ids,
        "cols": _mosaic_cols(len(ids)),          # kept for any other reader
        "tile": [_MOSAIC_TW, _MOSAIC_TH],        # kept
        "mode": lay.get("mode", "grid"),
        "mains": [c for c in (lay.get("mains") or []) if c in ids],
        "canvas": [cw, ch],
        "tiles": tiles,
    }



# ── Thread 1: muxer ──────────────────────────────────────────────────────────
# Drains every camera's queue continuously into a one-deep latest-frame slot,
# then on a FIXED tick snapshots all live slots and hands the whole set to the
# inference worker. It never runs the model, ByteTrack, or annotation, so it
# holds its 24 Hz tick regardless of how long a detection pass takes — which is
# what stops slow inference from backpressuring the camera WebSockets.
def _muxer_loop() -> None:
    tick_period = 1.0 / max(float(cfg["model"].get("stream_fps", 24.0)), 1.0)
    latest: dict[int, tuple[float, "np.ndarray"]] = {}
    configured = {int(s["id"]) for s in cfg.get("streams", [])}
    last_prune = time.monotonic()
    next_tick = time.monotonic()
    tick_window_start = time.monotonic()
    ticks = 0
    idle_logged = 0

    def drain() -> None:
        # cap.read() pops one frame; the inner loop drains any backlog and keeps
        # only the newest, so draining once per tick is equivalent to doing it
        # continuously — without the busy-poll that was starving other threads
        # (and the load-test client process) of CPU.
        for cam_id, cap in list(captures.items()):
            fd = cap.read()
            while fd is not None:
                latest[cam_id] = fd
                fd = cap.read()

    while True:
        if not captures:
            time.sleep(0.1)
            continue

        # Sleep until the tick is due (self-correcting accumulator), then drain.
        slack = next_tick - time.monotonic()
        if slack > 0:
            time.sleep(min(slack, tick_period))
        next_tick += tick_period
        if time.monotonic() - next_tick > 0.5:    # fell badly behind; resync
            next_tick = time.monotonic()
        drain()

        ticks += 1
        tw = time.monotonic() - tick_window_start
        if tw >= 1.0:
            _stats["mux_tps"] = ticks / tw
            ticks = 0
            tick_window_start = time.monotonic()

        # Prune dynamic devices (phones, replay) gone a while, so the dashboard
        # reflects who is actually here. Configured RTSP/webcam streams stay.
        if time.monotonic() - last_prune > 5.0:
            last_prune = time.monotonic()
            # Re-read every pass: streams added / removed through the API must
            # not be treated as the startup set (a camera added at runtime would
            # otherwise be pruned as a "dynamic device" after 20 s offline).
            configured = {int(s["id"]) for s in cfg.get("streams", [])}
            for cid in [c for c, cap in captures.items()
                        if c not in configured and not cap.socket_open
                        and (time.monotonic() - cap._last_frame_at) > 20.0]:
                captures.pop(cid, None)
                latest.pop(cid, None)
                _latest_bgr.pop(cid, None)
                _camera_meta.pop(cid, None)
                logger.info("CAM-%02d pruned from device list (gone > 20s)", cid)

        for cid in [c for c in latest if c not in captures]:
            latest.pop(cid, None)              # removed camera: stop feeding it
        now = time.monotonic()
        batch = [(cid, ts, frm) for cid, (ts, frm) in list(latest.items())
                 if (now - ts) < WebSocketCapture.STALE_AFTER]

        if not batch:
            idle_logged += 1
            if idle_logged == 300:
                connected = [cid for cid, c in captures.items() if c.connected]
                if connected:
                    logger.warning("No frames for ~%ds from connected camera(s) %s — "
                                   "the sender may have stalled",
                                   int(300 * tick_period), connected)
                else:
                    logger.info("Waiting for a camera to connect…")
                idle_logged = 0
            continue
        idle_logged = 0

        # Hand off. Depth-1 queue: if the worker is still busy, drop the stale
        # snapshot and enqueue the fresh one — process the newest frames, never
        # a backlog.
        try:
            _infer_q.put_nowait(batch)
        except queue.Full:
            try:
                _infer_q.get_nowait()
                _stats["dropped"] = _stats.get("dropped", 0) + 1
            except queue.Empty:
                pass
            try:
                _infer_q.put_nowait(batch)
            except queue.Full:
                pass


# ── Thread 2: inference worker ───────────────────────────────────────────────
# Owns the Detector (and therefore all per-camera ByteTrack / motion-gate
# state — single-threaded, no locks needed) plus risk scoring and evidence.
# Runs at its own pace: a slow pass just means it consumes fewer snapshots,
# the muxer keeps ticking, and video/detection degrade gracefully instead of
# stalling the ingest.
def _inference_worker() -> None:
    global _detector, _event_store, _harvester, _calibrator, _snapshots
    learn_cfg = cfg.get("learning", {}) or {}
    _calibrator = (Calibrator(cfg)
                   if learn_cfg.get("calibration", {}).get("enabled") else None)
    if _calibrator is not None:
        _restore_calibration(_calibrator)
    detector = Detector(cfg, num_cameras=0, calibrator=_calibrator,
                       native_frame_fn=_native_frame_fn)
    _detector = detector                      # expose for /status and /devices
    _stats["device"] = detector.device
    init_w = str(cfg.get("model", {}).get("weights", ""))
    _stats["active_model"] = "medium" if "26m" in init_w else "nano"
    _stats["switching"] = False
    risk_engine = RiskEngine(cfg)
    evidence = EvidenceChain(cfg["evidence"]["hash_chain_path"])
    store = EventStore(cfg["evidence"]["db_path"])
    _event_store = store
    geofence = GeoFenceEngine(cfg)
    geofence.load()
    gf_enabled = bool(cfg.get("geofence", {}).get("enabled"))
    gf_eval_on_track = cfg.get("geofence", {}).get("eval_on", "detect") == "track"
    snapshots = SnapshotWriter(cfg)
    _snapshots = snapshots
    _harvester = Harvester(cfg) if learn_cfg.get("enabled") else None
    if _harvester is not None:
        _harvester.set_active_model(_stats["active_model"])
    prev_levels: dict[int, str] = {}
    log_levels = {"High", "Critical"}
    _calib_last_decay = time.monotonic()

    win_start = time.monotonic()
    win_frames = 0
    win_batches = 0

    while True:
        # Pending fence-set reload from POST/DELETE /api/fences
        try:
            geofence.set_fences(_fences_q.get_nowait())
            logger.info("Geofence reloaded: %d fence(s)", geofence.count)
        except queue.Empty:
            pass

        # Continuous learning: operator review verdicts + calibration commands
        for _ in range(64):
            try:
                item = _learn_q.get_nowait()
            except queue.Empty:
                break
            if _harvester is not None:
                _harvester.apply_review(item.get("id", ""), item.get("verdict", ""),
                                        item.get("boxes"))
        try:
            if _calibrator is not None:
                _calibrator.load(_calib_q.get_nowait())
                _persist_calibration(_calibrator)
        except queue.Empty:
            pass
        if _calibrator is not None and time.monotonic() - _calib_last_decay > 3600:
            _calib_last_decay = time.monotonic()
            _calibrator.decay()
            _persist_calibration(_calibrator)

        # Check for pending model switch — str profile name, or dict weight paths
        try:
            item = _switch_q.get_nowait()
            logger.info("Hot-swapping model: %s", item)
            _stats["switching"] = True
            old_det = detector
            try:
                if isinstance(item, dict):
                    new_det = Detector.from_weights(
                        cfg, item["weights"], item["pose_weights"],
                        label=item.get("label", ""), num_cameras=0,
                        anpr=getattr(old_det, "anpr", None), calibrator=_calibrator,
                        weapon=getattr(old_det, "weapon", None),
                        vehicle_type=getattr(old_det, "vehicle_type", None),
                        anpr_results=getattr(old_det, "anpr_results", None),
                        vehicle_arrival=getattr(old_det, "_vehicle_arrival", None),
                        native_frame_fn=getattr(old_det, "_native_frame_fn", None))
                    _stats["active_model"] = item.get("name", "ft")
                else:
                    new_det = Detector.from_profile(
                        cfg, item, num_cameras=0,
                        anpr=getattr(old_det, "anpr", None), calibrator=_calibrator,
                        weapon=getattr(old_det, "weapon", None),
                        vehicle_type=getattr(old_det, "vehicle_type", None),
                        anpr_results=getattr(old_det, "anpr_results", None),
                        vehicle_arrival=getattr(old_det, "_vehicle_arrival", None),
                        native_frame_fn=getattr(old_det, "_native_frame_fn", None))
                    _stats["active_model"] = item
                detector = new_det
                _detector = detector
                _stats["device"] = detector.device
                if _harvester is not None:
                    _harvester.set_active_model(_stats["active_model"])
                logger.info("Model hot-swapped to '%s' on %s",
                            _stats["active_model"], detector.device)
                del old_det
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error("Model switch failed (%r): %s", item, e)
                detector = old_det
            finally:
                _stats["switching"] = False
        except queue.Empty:
            pass

        try:
            batch = _infer_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if not batch:
            continue

        results = detector.process_batch(batch)
        _stats["frames"] += len(batch)
        win_frames += len(batch)
        win_batches += 1


        elapsed = time.monotonic() - win_start
        if elapsed >= 1.0:
            ds = detector.stats
            _stats["fps"] = win_frames / elapsed
            _stats["worker_bps"] = win_batches / elapsed
            _stats["infer_ms"] = ds.infer_ms
            _stats["mean_batch"] = ds.mean_batch
            _stats["last_batch"] = ds.last_batch
            _stats["gpu_saving_pct"] = ds.gpu_saving_pct
            _stats["frames_inferred"] = ds.frames_inferred
            _stats["frames_gated"] = ds.frames_gated
            _stats["frames_rate_skipped"] = ds.frames_rate_skipped
            _stats["detector_passes"] = ds.detector_passes
            logger.info(
                "PIPE  %d cams | mux %.1f tps | worker %.0f fps (%.1f bps) | "
                "batch %.1f | %.1f ms/frame | GPU saw %d of %d (%.0f%% skipped) | "
                "dropped %d | %s",
                len(captures), _stats.get("mux_tps", 0.0), _stats["fps"],
                _stats["worker_bps"], ds.mean_batch, ds.infer_ms,
                ds.frames_inferred, ds.frames_in, ds.gpu_saving_pct,
                _stats.get("dropped", 0), _stats.get("device", "?"),
            )
            win_start = time.monotonic()
            win_frames = 0
            win_batches = 0

        frames_by_cam: dict = {}
        raw_by_cam: dict = {}                 # cam_id -> raw frame, for the weapon snapshot
        if _harvester is not None:
            for sr in results:
                _harvester.submit(sr.cam_id, sr.frame, sr.detections,
                                  sr.inferred, sr.timestamp)
                if sr.inferred:
                    frames_by_cam[sr.cam_id] = sr.frame
                    for d in sr.detections:
                        if d.is_person and d.posture is not None:
                            _harvester.submit_posture(
                                sr.cam_id, sr.timestamp, f"{sr.cam_id}_{d.track_id}",
                                getattr(d.posture, "keypoints", None), d.posture.label)
                    if sr.weapons:
                        _harvester.submit_weapon(sr.cam_id, sr.frame, sr.weapons,
                                                 sr.timestamp)

        for sr in results:
            ra = risk_engine.assess(sr)
            cap = captures.get(sr.cam_id)
            if cap is None:
                # Removed while this batch was being inferred — publishing it
                # would put a frozen tile back on the mosaic.
                continue
            latency_ms = int(cap.latency_ms) if cap is not None else 0
            snapshots.ensure_cam(sr.cam_id)
            live_ids = {d.track_id for d in sr.detections if d.track_id >= 0}

            # ── Virtual-fence breach → CRITICAL (mirrors the weapon override) ──
            new_breaches = []
            if gf_enabled and (sr.inferred or gf_eval_on_track):
                new_breaches = geofence.evaluate(sr)
                geofence.prune(sr.cam_id, live_ids)
            if sr.inferred:
                snapshots.prune(sr.cam_id, live_ids)
            zones = geofence.active_zones(sr.cam_id)
            _geofence_active[sr.cam_id] = zones
            _geofence_fences[sr.cam_id] = geofence.public_for(sr.cam_id)
            breach_now = any(zones.values())
            breach_zones = [f.label or f.id
                            for f in geofence.fences_for(sr.cam_id) if zones.get(f.id)]
            if breach_now:
                ra.level, ra.score = "Critical", max(ra.score, 90.0)

            # ── Overlay FIRST, so any snapshot taken below carries the boxes,
            #    skeleton, fence lines and banner. NO JPEG encoding here — the
            #    /stream endpoints encode in their own worker threads. ─────────
            frame_out = sr.annotated_frame if sr.annotated_frame is not None else sr.frame
            colour = ((0, 200, 0) if ra.level == "Normal"
                      else (0, 165, 255) if ra.level == "High" else (0, 0, 255))
            tag = "LIVE" if sr.inferred else "trk"
            label = (f"CAM-{sr.cam_id:02d}  {ra.level} {ra.score:.0f}  "
                     f"P:{sr.person_count} V:{sr.vehicle_count}  [{tag}]")
            _draw_label(frame_out, label, (8, 24), 0.55, colour)
            _draw_fences(frame_out, _geofence_fences.get(sr.cam_id, []), zones)
            if latency_ms:
                late = latency_ms >= 1500
                txt = (f"NET {latency_ms/1000:.1f}s  DELAYED" if late
                       else f"NET {latency_ms} ms")
                scale = 0.62 if late else 0.5
                (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
                _draw_label(frame_out, txt, (frame_out.shape[1] - tw - 12, 24),
                            scale, (0, 0, 255) if late else (0, 210, 120))
            if ra.weapon_tier == "posture" and not ra.weapon:
                _draw_banner(frame_out, "AIM POSTURE (unconfirmed)  -  HIGH")
            elif ra.weapon:
                _draw_banner(frame_out, {
                    "armed":   "ARMED THREAT  -  CRITICAL",
                    "gun":     "GUN DETECTED  -  CRITICAL",
                    "posture": "WEAPON (posture)  -  CRITICAL",
                }.get(ra.weapon_tier, "WEAPON  -  CRITICAL"))
            _latest_bgr[sr.cam_id] = frame_out
            raw_by_cam[sr.cam_id] = sr.frame

            # ── Fence breach → hash-chained evidence + a snapshot of the exact
            #    moment (data/snapshots/<cam_id>/). The image sha256 is embedded
            #    in the breach record, so the file is tamper-evident. ──────────
            for b in new_breaches:
                snap = None
                if snapshots.should_capture(sr.cam_id, b.track_id, "breach"):
                    snap = snapshots.capture(
                        sr.cam_id, frame_out, sr.frame, "breach",
                        {"detail": b.fence.label or b.fence.id, "track_id": b.track_id})
                bev = {"cam_id": sr.cam_id, "type": "breach",
                       "fence_id": b.fence.id, "fence_label": b.fence.label,
                       "fence_kind": b.fence.kind, "track_id": b.track_id,
                       "class_name": b.class_name, "direction": b.direction,
                       "ground_point": [round(x, 4) for x in b.ground_point],
                       "level": "Critical",
                       "evidence_file": (snap or {}).get("file"),
                       "sha256": (snap or {}).get("sha256")}
                bh = evidence.append(bev)
                store.log(sr.cam_id, "Breach", ra.score,
                          sr.person_count, sr.vehicle_count, bev, bh)
                if snap is not None:
                    snap["ev_hash"] = bh
                    snapshots.record(snap)
                logger.warning("CAM-%02d BREACH  fence=%s  track=%d  %s  hash=%s…",
                               sr.cam_id, b.fence.label or b.fence.id,
                               b.track_id, b.fence.kind, bh[:12])

            # ── Malicious posture (crouch / lying) → snapshot + evidence at
            #    "Posture" severity. Does NOT force Critical unless
            #    snapshots.escalate_posture is set (config.yaml). ──────────────
            if snapshots.wants("posture") and sr.inferred:
                for d in malicious_postures(sr.detections):
                    if not snapshots.should_capture(sr.cam_id, d.track_id, "posture"):
                        continue
                    snap = snapshots.capture(
                        sr.cam_id, frame_out, sr.frame, "posture",
                        {"detail": d.posture.label, "track_id": d.track_id})
                    pev = {"cam_id": sr.cam_id, "type": "posture",
                           "posture": d.posture.label, "track_id": d.track_id,
                           "level": "High",
                           "evidence_file": (snap or {}).get("file"),
                           "sha256": (snap or {}).get("sha256")}
                    ph = evidence.append(pev)
                    store.log(sr.cam_id, "Posture", ra.score,
                              sr.person_count, sr.vehicle_count, pev, ph)
                    if snap is not None:
                        snap["ev_hash"] = ph
                        snapshots.record(snap)
                    logger.warning("CAM-%02d MALICIOUS POSTURE %s  track=%d  hash=%s…",
                                   sr.cam_id, d.posture.label or "?", d.track_id, ph[:12])
                    if snapshots.escalate_posture and ra.level != "Critical":
                        ra.level, ra.score = "Critical", max(ra.score, 90.0)

            prev = prev_levels.get(sr.cam_id, "Normal")
            if ra.level != prev:
                prev_levels[sr.cam_id] = ra.level
                if ra.level in log_levels:
                    event = {
                        "cam_id": sr.cam_id, "level": ra.level,
                        "score": round(ra.score, 2),
                        "persons": sr.person_count, "vehicles": sr.vehicle_count,
                    }
                    ev_hash = evidence.append(event)
                    store.log(sr.cam_id, ra.level, ra.score,
                              sr.person_count, sr.vehicle_count, event, ev_hash)
                    logger.warning("CAM-%02d %s  score=%.1f  hash=%s…",
                                   sr.cam_id, ra.level, ra.score, ev_hash[:12])

            _camera_meta[sr.cam_id] = {
                "cam_id": sr.cam_id,
                "level": ra.level,
                "score": round(ra.score, 1),
                "persons": sr.person_count,
                "vehicles": sr.vehicle_count,
                "weapon": ra.weapon,
                "armed": ra.armed,
                "weapon_tier": ra.weapon_tier,
                "weapon_boxes": [
                    [*h.bbox, round(h.conf, 2), bool(h.confirmed)]
                    for h in (sr.weapons or [])
                ],
                "latency_ms": latency_ms,
                "anomalies": [
                    d.posture.label
                    for d in sr.detections
                    if d.is_person and d.posture is not None and d.posture.any_anomaly
                ],
                "plates": [
                    {"track": d.track_id, "text": d.plate,
                     "conf": round(d.plate_conf, 2)}
                    for d in sr.detections if d.is_vehicle and d.plate
                ],
                "breach": breach_now,
                "breach_zones": breach_zones,
            }

        # Confirmed number plates → the hash-chained evidence log, and — the
        # dedicated ANPR output folder (data/anpr_results/, separate from
        # data/snapshots/) — complete the pending arrival record this plate
        # belongs to with its text/confidence.
        if detector.anpr is not None:
            for pr in detector.anpr.drain_events():
                ev = {"cam_id": pr.cam_id, "type": "plate", "plate": pr.text,
                      "confidence": round(pr.conf, 2), "verified": pr.valid,
                      "track": pr.track_id}
                h = evidence.append(ev)
                store.log(pr.cam_id, "Plate", pr.conf * 100, 0, 1, ev, h)
                if _harvester is not None:
                    _harvester.submit_plate(pr.cam_id, frames_by_cam.get(pr.cam_id), pr)
                if detector.anpr_results is not None:
                    arec = detector.anpr_results.finalize(pr.cam_id, pr.track_id, pr)
                    if arec is not None:
                        if detector.anpr_results.log_to_evidence:
                            arec["ev_hash"] = evidence.append({
                                "cam_id": arec["cam_id"], "type": "anpr_result",
                                "track_id": arec["track_id"], "plate": arec["plate"],
                                "plate_conf": arec["plate_conf"],
                                "vehicle_type": arec["vehicle_type"],
                                "vehicle_type_conf": arec["vehicle_type_conf"],
                                "evidence_file": arec["file"], "sha256": arec["sha256"]})
                        detector.anpr_results.record(arec)

        # Vehicle arrivals whose plate never resolved (unreadable / no plate
        # visible) still get filed — the picture + vehicle type are useful on
        # their own — after anpr_results.finalize_timeout_s.
        if detector.anpr_results is not None:
            for arec in detector.anpr_results.sweep_timeouts():
                if detector.anpr_results.log_to_evidence:
                    arec["ev_hash"] = evidence.append({
                        "cam_id": arec["cam_id"], "type": "anpr_result",
                        "track_id": arec["track_id"], "plate": arec["plate"],
                        "plate_conf": arec["plate_conf"],
                        "vehicle_type": arec["vehicle_type"],
                        "vehicle_type_conf": arec["vehicle_type_conf"],
                        "evidence_file": arec["file"], "sha256": arec["sha256"]})
                detector.anpr_results.record(arec)

        # Confirmed firearm detections → the hash-chained evidence log (same
        # override shape as a fence breach), plus a snapshot of the moment. The
        # annotated frame (_latest_bgr) already carries the GUN banner drawn
        # above. "armed" escalation is logged by the level-transition block.
        if getattr(detector, "weapon", None) is not None:
            for we in detector.weapon.drain_events():
                snap = None
                if snapshots.should_capture(we.cam_id, we.track_id, "weapon"):
                    snap = snapshots.capture(
                        we.cam_id, _latest_bgr.get(we.cam_id),
                        raw_by_cam.get(we.cam_id), "weapon",
                        {"detail": we.tier, "track_id": we.track_id})
                ev = {"cam_id": we.cam_id, "type": "weapon", "tier": we.tier,
                      "confidence": round(we.conf, 2), "track": we.track_id,
                      "bbox": [int(v) for v in we.bbox],
                      "evidence_file": (snap or {}).get("file"),
                      "sha256": (snap or {}).get("sha256")}
                h = evidence.append(ev)
                store.log(we.cam_id, "Critical", 94.0, 0, 0, ev, h)
                if snap is not None:
                    snap["ev_hash"] = h
                    snapshots.record(snap)
                logger.warning("CAM-%02d GUN CONFIRMED  track=%d  conf=%.0f%%  hash=%s...",
                               we.cam_id, we.track_id, we.conf * 100, h[:12])


def _draw_label(img, text: str, org, scale: float, colour) -> None:
    """Text with a black outline so it reads on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, 1, cv2.LINE_AA)


def _draw_banner(img, text: str) -> None:
    """A full-width red alert bar across the middle of the frame."""
    h, w = img.shape[:2]
    y0, y1 = int(h * 0.42), int(h * 0.58)
    strip = img[y0:y1].copy()
    cv2.rectangle(strip, (0, 0), (w, y1 - y0), (0, 0, 200), -1)
    cv2.addWeighted(strip, 0.55, img[y0:y1], 0.45, 0, img[y0:y1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    org = ((w - tw) // 2, (y0 + y1) // 2 + th // 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_fences(img, fences: list, active: dict) -> None:
    """Draw virtual fences onto an annotated frame — green normally, solid red
    (filled for polygons) while a fence is being breached. Runs every worker
    pass so fences show on the mosaic even with the drawing UI closed."""
    if not fences:
        return
    h, w = img.shape[:2]
    for f in fences:
        pts = np.array([(int(x * w), int(y * h)) for x, y in f["points"]], np.int32)
        if len(pts) < 2:
            continue
        hot = bool(active.get(f["id"]))
        col = (0, 0, 255) if hot else (60, 200, 60)
        closed = f["kind"] == "polygon"
        if hot and closed:
            ov = img.copy()
            cv2.fillPoly(ov, [pts], col)
            cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)
        cv2.polylines(img, [pts], closed, col, 2, cv2.LINE_AA)
        if not closed and f.get("direction", "both") != "both":
            # a small arrow on every segment, along the segment normal, pointing
            # to the side a target ends up on for an "a2b" crossing
            sign = 1.0 if f["direction"] == "a2b" else -1.0
            for a, b in zip(pts[:-1], pts[1:]):
                mid = (a + b) / 2.0
                seg = b - a
                nrm = (seg[0] ** 2 + seg[1] ** 2) ** 0.5 or 1.0
                perp = np.array([seg[1], -seg[0]]) / nrm * sign   # right of A→B
                tip = (mid + perp * 20).astype(int)
                cv2.arrowedLine(img, tuple(mid.astype(int)), tuple(tip),
                                col, 2, cv2.LINE_AA, tipLength=0.4)
        label = f.get("label") or f["id"]
        _draw_label(img, label, (int(pts[:, 0].min()) + 3,
                                 max(int(pts[:, 1].min()) - 4, 12)), 0.45, col)


def _mosaic_cols(n: int) -> int:
    return 1 if n == 1 else 2 if n <= 4 else 3 if n <= 9 else 4


_MOSAIC_TW, _MOSAIC_TH = 480, 270          # base tile size (canvas pixels)


def _mosaic_tiles(ids: list) -> tuple[int, int, list[dict]]:
    """Rect map for the current mosaic layout.

    Returns ``(canvas_w, canvas_h, tiles)`` where each tile is
    ``{cam_id, x, y, w, h}`` in canvas pixels. Two layouts, chosen by
    ``_stats["layout"]`` (set via POST /api/layout — the mosaic is one shared
    composition, so the layout is global):

      * ``grid``  — uniform packing, row-major by sorted cam_id (the classic).
      * ``focus`` — 1-4 "main" cameras fill the left area large; every other
        camera is a small tile down a right-side filmstrip.

    The dashboard maps a mouse click on the mosaic back to a camera through
    this exact rect map (see /status.mosaic.tiles), so grid is just the special
    case where every rect is equal.
    """
    ids = list(ids)
    tw, th = _MOSAIC_TW, _MOSAIC_TH
    lay = _stats.get("layout", {}) or {}
    mode = lay.get("mode", "grid")
    mains = [c for c in (lay.get("mains") or []) if c in ids][:4]

    if mode != "focus" or not mains:
        n = len(ids) or 1
        cols = _mosaic_cols(n)
        rows = (n + cols - 1) // cols
        tiles = [{"cam_id": cid, "x": (i % cols) * tw, "y": (i // cols) * th,
                  "w": tw, "h": th} for i, cid in enumerate(ids)]
        return cols * tw, rows * th, tiles

    rest = [c for c in ids if c not in mains]
    m = len(mains)
    mcols = 1 if m == 1 else 2
    mrows = 1 if m <= 2 else 2
    main_w, main_h = mcols * tw, mrows * th
    strip_w = tw // 2 if rest else 0
    canvas_w, canvas_h = main_w + strip_w, main_h

    mtw, mth = main_w // mcols, main_h // mrows
    tiles = [{"cam_id": cid, "x": (i % mcols) * mtw, "y": (i // mcols) * mth,
              "w": mtw, "h": mth} for i, cid in enumerate(mains)]
    if rest:
        sth = max(1, canvas_h // len(rest))
        tiles += [{"cam_id": cid, "x": main_w, "y": j * sth,
                   "w": strip_w, "h": sth} for j, cid in enumerate(rest)]
    return canvas_w, canvas_h, tiles


def _build_mosaic() -> bytes | None:
    """Tile every camera's latest annotated frame into one JPEG per the current
    layout. Runs in a request worker thread (see /stream), never on the
    detection loop.
    """
    ids = sorted(_latest_bgr)
    if not ids:
        return None
    cw, ch, tiles = _mosaic_tiles(ids)
    grid = np.zeros((ch, cw, 3), dtype=np.uint8)
    for t in tiles:
        f = _latest_bgr.get(t["cam_id"])
        if f is None or t["w"] < 2 or t["h"] < 2:
            continue
        grid[t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"]] = \
            cv2.resize(f, (t["w"], t["h"]))
    ok, jpeg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes() if ok else None


# ── Entry point ───────────────────────────────────────────────────────────────
# Two listeners share the same app and detection state:
#   HTTPS :8443 — phones (getUserMedia demands a secure context)
#   HTTP  :8090 — monitor dashboard on this machine (no cert warning)
HTTPS_PORT = 8443
HTTP_PORT = 8090      # 8080 is commonly taken (Steam webhelper, Jenkins, Tomcat)


def _lan_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))     # no packets sent; just picks the route
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _port_taken(port: int) -> bool:
    """Windows lets two sockets share a port, so probe for a live responder."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.4)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


_MODE_MENU = """
+-- IBVAP -- input source  ->  where to render --------------------
|  [1] Screen  ->  DASHBOARD        (analysis on the web UI, :8090/monitor)
|  [2] Screen  ->  ON-SCREEN OVERLAY (transparent boxes drawn on the screen)
|  [3] Phone camera  ->  dashboard
|  [4] Real cameras (config.yaml RTSP)  ->  dashboard
|  [5] Everything in config.yaml   ->  dashboard   (default)
+---------------------------------------------------------------
Select [1-5], Enter for 5: """
# choice -> (input mode, render surface)
_MENU_CHOICE = {
    "1": ("screen", "dashboard"),
    "2": ("screen", "overlay"),
    "3": ("phone", "dashboard"),
    "4": ("cctv", "dashboard"),
    "5": ("all", "dashboard"),
    "":  ("all", "dashboard"),
}


def _choose_mode() -> tuple[str, str]:
    """Interactive picker for `python server.py` with no --mode. Returns
    (input_mode, render_surface). Falls back to ('all','dashboard') when stdin
    is not a TTY (run_demo.py, CI, nohup)."""
    if not sys.stdin.isatty():
        return "all", "dashboard"
    try:
        choice = input(_MODE_MENU).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "all", "dashboard"
    return _MENU_CHOICE.get(choice, ("all", "dashboard"))


if __name__ == "__main__":
    import argparse
    import uvicorn
    from pathlib import Path as _Path

    _ap = argparse.ArgumentParser(description="IBVAP web server")
    _ap.add_argument("--mode", choices=["menu", "screen", "phone", "cctv", "all"],
                     default="menu",
                     help="input source: screen grab / phone slots / pulled CCTV / "
                          "everything in config.yaml. 'menu' (default) prompts when "
                          "run interactively, else 'all'.")
    _ap.add_argument("--surface", choices=["dashboard", "overlay"], default="dashboard",
                     help="screen mode only — where to render: 'dashboard' (web UI) "
                          "or 'overlay' (transparent boxes drawn on the screen; this "
                          "hands off to screen_watch.py).")
    _ap.add_argument("--region", default=None,
                     help="screen mode: X,Y,W,H pixels to grab (default: whole monitor)")
    _ap.add_argument("--monitor", type=int, default=1,
                     help="screen mode: monitor index (1 = primary)")
    _args = _ap.parse_args()

    if _args.mode == "menu":
        _STARTUP_MODE, _SURFACE = _choose_mode()
    else:
        _STARTUP_MODE, _SURFACE = _args.mode, _args.surface
    _SCREEN_MONITOR = _args.monitor
    if _args.region:
        try:
            _SCREEN_REGION = tuple(int(v) for v in _args.region.split(","))
            assert len(_SCREEN_REGION) == 4
        except (ValueError, AssertionError):
            logger.error("--region must be X,Y,W,H integers; ignoring")
            _SCREEN_REGION = None

    # Screen -> on-screen overlay is a different renderer (screen_watch.py paints
    # transparent boxes over the monitor); the FastAPI server is not involved.
    if _STARTUP_MODE == "screen" and _SURFACE == "overlay":
        import subprocess
        _sw = [sys.executable, "scripts/screen_watch.py"]
        if _args.region:
            _sw += ["--region", _args.region]
        if _args.monitor and _args.monitor != 1:
            _sw += ["--monitor", str(_args.monitor)]
        logger.info("Screen -> ON-SCREEN OVERLAY: launching %s  (Esc on the overlay to stop)",
                    " ".join(_sw[1:]))
        raise SystemExit(subprocess.call(_sw, cwd=str(_Path(__file__).resolve().parent)))

    ip = _lan_ip()
    have_cert = _Path("cert.pem").exists() and _Path("key.pem").exists()

    # cfg itself loads in _lifespan, after the sockets are bound — the network
    # section has to be read here.
    try:
        _net = (yaml.safe_load(open("config.yaml", encoding="utf-8")) or {}).get("network") or {}
    except (OSError, yaml.YAMLError):
        _net = {}
    _BIND_HOST = str(_net.get("bind_host", "127.0.0.1"))
    _LAN_PHONE_INTAKE = bool(_net.get("lan_phone_intake", False)) and have_cert
    # The HTTPS listener exists only for phone intake — directly on the LAN or
    # through a "phones only" Cloudflare tunnel (cloudflared -> :8443). Always
    # restrict it to the camera routes so neither path can reach the API.
    if have_cert:
        app.add_middleware(ListenerGuard, lan_port=HTTPS_PORT)
    _stats["lan_phone_intake"] = _LAN_PHONE_INTAKE

    for _p, _label in ((HTTP_PORT, "HTTP"), (HTTPS_PORT, "HTTPS")):
        if _port_taken(_p):
            logger.error("Port %d (%s) is already answering — another process owns it. "
                         "Stop it, or change the port in server.py.", _p, _label)
            raise SystemExit(1)

    logger.info("+-- IBVAP ------------------------------------------------")
    logger.info("|  INPUT MODE: %s", _STARTUP_MODE.upper())
    logger.info("|  MONITOR (this PC, no cert warning)")
    logger.info("|     http://localhost:%d/monitor", HTTP_PORT)
    if _STARTUP_MODE == "screen":
        logger.info("|  Grabbing the screen as CAM-00 — open the monitor above.")
    elif have_cert and _STARTUP_MODE in ("phone", "all"):
        logger.info("|  CAMERA (phone, accept the certificate once)")
        logger.info("|     https://%s:%d/camera/0", ip, HTTPS_PORT)
        logger.info("|     https://%s:%d/camera/1   (second phone)", ip, HTTPS_PORT)
    elif not have_cert and _STARTUP_MODE in ("phone", "all"):
        logger.warning("|  No cert.pem — phone cameras need HTTPS. Run: python tools/gen_cert.py")
    if _STARTUP_MODE == "cctv":
        logger.info("|  Pulling RTSP/CCTV streams from config.yaml `streams:`.")
    logger.info("|  STATUS   http://localhost:%d/status", HTTP_PORT)
    logger.info("|  NETWORK  API/dashboard/streams bound to %s (%s)", _BIND_HOST,
                "local only" if _BIND_HOST in ("127.0.0.1", "localhost", "::1")
                else "EXPOSED BEYOND THIS MACHINE")
    if _LAN_PHONE_INTAKE:
        logger.warning("|  TEST MODE: phone intake open to the LAN on :%d "
                       "(camera pages only). Set network.lan_phone_intake: false "
                       "after the demo.", HTTPS_PORT)
    elif have_cert and _STARTUP_MODE in ("phone", "all"):
        logger.warning("|  Phone intake is local-only (network.lan_phone_intake: false) "
                       "- phones on the LAN cannot connect.")
    logger.info("+---------------------------------------------------------")

    # proxy_headers + forwarded_allow_ips: when the server sits behind a tunnel
    # (tunnel.py -> cloudflared / ngrok), trust the X-Forwarded-Proto/-For it
    # sets so request.url.scheme is "https" and client IPs are the real ones.
    # Safe here because the only untunnelled exposure is the LAN.
    # Only a proxy on this machine may rewrite the client address. "*" let any
    # LAN host send X-Forwarded-For: 127.0.0.1 and pass as loopback, which
    # would defeat the write guard (ibvap/security.py).
    _common = dict(log_level="info", access_log=False,
                   proxy_headers=True, forwarded_allow_ips="127.0.0.1")

    async def _serve() -> None:
        servers = [
            uvicorn.Server(uvicorn.Config(
                app, host=_BIND_HOST, port=HTTP_PORT, **_common,
            ))
        ]
        if have_cert:
            servers.append(uvicorn.Server(uvicorn.Config(
                app, host="0.0.0.0" if _LAN_PHONE_INTAKE else _BIND_HOST,
                port=HTTPS_PORT,
                ssl_certfile="cert.pem", ssl_keyfile="key.pem", **_common,
            )))
        await asyncio.gather(*(s.serve() for s in servers))

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        logger.info("Shutting down")
