# IBVAP — AI-Based Intelligent Video Analytics Platform for Border Surveillance

SIH 2026 · PS 26187 · Ministry of Home Affairs / Sashastra Seema Bal (SSB)

A software-only analytics layer on top of existing CCTV. Multiple camera
streams (RTSP or phones via a browser link — no app install) are batched
through one **YOLO26n** detector on a single GPU, tracked with **ByteTrack**,
scored for risk, and shown on a live command dashboard. Events are written to a
SHA-256 hash-chained evidence log.

## Highlights

- **Batched multi-stream inference** — every camera's frame goes through the
  model in one forward pass (the nvstreammux idea). Measured 8×720p@24fps at
  ~192 fps aggregate on an RTX 3050 6 GB, GPU ~35 % — from phone / pre-loaded
  sources, so it excludes H.264 decode; real-RTSP camera counts are unmeasured.
- **Two-thread pipeline** — a fixed-tick muxer feeds a separate inference
  worker, so a slow GPU pass never backpressures the camera sockets.
- **Motion gating** — a cheap frame-difference pre-filter; a static scene never
  reaches the GPU (~80 % of frames skipped on real footage).
- **TensorRT FP16 engine** (`yolo26n.engine`) — the default backend, 4.8× the
  batch-1 latency of PyTorch.
- **Real CCTV ingestion** — RTSP / NVR channels / ONVIF cameras pulled on their
  own decode threads (TCP, auto-reconnect, stall watchdog), in the same process
  and pipeline as the phones. `tools/rtsp_probe.py` + `tools/discover_cameras.py` to set up.
- **Phone ingestion with latency control** — browser `getUserMedia` → JPEG over
  WebSocket, adaptive to uplink backpressure, with a per-stream latency badge.
- **Posture rules** — lying / crouch / scan / arms-up / two-handed weapon-ready
  grip (the last forces a Critical alert).
- **Weapon detection** — a dedicated YOLO `gun` model (`models/weapon_detector.pt`,
  trained by `training/train_weapon.py` on [YouTube-GDD](https://github.com/UCAS-GYX/YouTube-GDD))
  runs as one extra batched pass. A gun is alerted only when *held* by a tracked
  person and persistent; `AIM` posture **+** a confirmed gun on one track escalates
  to `ARMED THREAT` (score 99), written to the hash-chained evidence log. See
  `docs/WEAPON_DETECTION.md`.
- **ANPR** — YOLOv8 plate detector on vehicle crops + threaded EasyOCR, Indian
  plate-format cleanup, logged to a CSV and the hash-chained evidence trail.
  (Ported from [anindya-mukhopadhyay/ANPR](https://github.com/anindya-mukhopadhyay/ANPR), MIT.)
- **Virtual fences** — draw polygon zones and directional tripwire lines on each
  camera view in the dashboard; a tracked target that breaches one is forced
  Critical and written to the hash-chained evidence trail. Image-space (no camera
  calibration exists to geo-project), reloaded live. See `docs/GEOFENCE.md`.
- **Off-LAN access** — `scripts/tunnel.py` (ngrok or cloudflared), including your own
  domain via a cloudflared named tunnel (see `docs/CUSTOM_DOMAIN.md`).
- **Screen-share overlay** (demo) — `scripts/screen_watch.py` points the same detector +
  pose pass at your laptop screen and paints boxes / skeletons back as a
  transparent click-through overlay. See `docs/SCREEN_WATCH.md`.
- **Launch-time source + render menu** — `python server.py` prompts for the
  input (**screen** / **phone** / **real cameras** / everything in config) and,
  for screen, *where to render*: **[1] dashboard** (analysis on the web UI) or
  **[2] on-screen overlay** (`scripts/screen_watch.py` — the full stack, detector + pose
  + gun detection + ANPR, painted transparently on the screen itself).
  Non-interactive:
  `python server.py --mode screen --surface dashboard|overlay`
  (or `--mode phone|cctv|all`).
- **Continuous learning** — the pipeline auto-labels its confident, track-stable
  detections into a review pool; the operator confirms/discards in the dashboard
  LEARNING panel; `python training/retrain.py` fine-tunes YOLO26n on the kept set and
  registers a new model you hot-swap in (with a before/after mAP) and roll back.
  Per-camera calibration also tunes confidence floors + suppresses recurring
  false-positive regions live, without training. See `docs/CONTINUOUS_LEARNING.md`.
- **Night / thermal model** — `python training/train_thermal.py` fine-tunes YOLO26n on
  infrared surveillance imagery ([LLVIP](https://github.com/bupt-ai-cz/LLVIP),
  optionally + FLIR ADAS) and registers it in the same model row. The operator
  swaps to **Thermal / night** for IR cameras and back to **Nano** for daytime —
  no restart. See `docs/THERMAL_NIGHT.md`.

## Quick start

```bash
# 1. PyTorch with CUDA 12.6, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. one-time assets
python tools/gen_cert.py            # self-signed cert for phone camera access on the LAN
python tools/make_test_videos.py    # 4× 720p/24fps demo clips
python tools/export_engine.py       # TensorRT engine for THIS GPU (optional; falls back to .pt)

# 3. run the demo — server + 4 replayed streams + dashboard
python run_demo.py --cams 4
#   dashboard  → http://localhost:8090/monitor
#   phone (LAN)→ https://<lan-ip>:8443/cam/0
#   phone (any network) → add --tunnel, open the URL it prints
```

## Layout

```
server.py · main.py · run_demo.py   entry points  (web server · OpenCV grid · one-command demo)
config.yaml                         streams, model, throughput, risk, pose/weapon/geofence thresholds

ibvap/            the pipeline package
  detector.py       batched YOLO26n + per-camera ByteTrack + track carry-forward + pose/anpr/weapon passes
  rtsp_capture.py · ws_capture.py · screen_capture.py   frame sources (CCTV · phone WS · screen grab)
  motion_gate.py    frame-difference pre-filter in front of the GPU
  posture.py        yolo26n-pose on person crops + geometry rules (LYING / CROUCH / AIM …)
  weapon.py         trained `gun` detector — held-gun confirmation, fused with AIM in risk_engine
  anpr.py           number-plate recognition — plate detector on vehicle crops + threaded EasyOCR
  geofence.py       virtual fences — polygon zones / tripwire lines, image-space breach detection
  risk_engine.py    zone × time × behaviour → 0–100 score (+ weapon / fence overrides)
  learn.py · calibration.py   continuous learning — harvest pool · per-camera live calibration
  evidence.py · event_store.py   SHA-256 hash chain · SQLite store-and-forward

scripts/          operational CLIs   run_demo helpers + measurement
  screen_watch.py   transparent on-screen overlay (also `server.py --mode screen --surface overlay`)
  feed_test.py · loadtest_mobile.py · benchmark.py · diagnose.py   replay / load / measure / health-check
  tunnel.py         off-LAN phone access (ngrok / cloudflared)

training/         model training + datasets  (weights land in models/, datasets in E:/Projects)
  datasets_fetch.py   pull YouTube-GDD (--weapon) / LLVIP + FLIR ADAS (--thermal) into E:/Projects
  train_weapon.py     fine-tune yolo26n → models/weapon_detector.pt
  train_thermal.py    fine-tune yolo26n for night/IR → models/registry.json "Thermal / night"
  retrain.py          continuous-learning fine-tune from the reviewed harvest pool

tools/            one-off setup / probes
  gen_cert.py · make_test_videos.py · export_engine.py   one-time assets (TLS cert · demo clips · TensorRT engine)
  rtsp_probe.py · discover_cameras.py   validate a camera URL · find ONVIF cameras on the LAN
  nms_check.py · test_detect.py         verify the NMS-free path · quick single-file detection

docs/ · notebooks/ · tests/ · static/ · models/
```

## Connecting real CCTV

```bash
python tools/discover_cameras.py --user admin --pass <pw>     # find ONVIF cameras on the LAN
python tools/rtsp_probe.py "rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/102"   # test one URL
```

Then add the (sub-stream) URLs to `config.yaml` `streams:` and run
`python server.py`. Full guide — how CCTV/NVRs are wired, vendor URL tables,
what to ask college IT for — is **`docs/CCTV_INTEGRATION.md`**.

## Deep docs

**[`CLAUDE.md`](CLAUDE.md) has the full architecture, measured numbers, and
design rationale.** Per-feature guides and the setup walkthrough are indexed in
**[`docs/README.md`](docs/README.md)** (start with
[`docs/STARTUP.md`](docs/STARTUP.md) on a fresh machine).

Model weights, the TensorRT engine, generated test videos, the TLS cert and
runtime data are `.gitignore`d — regenerate them with the commands above.

## Constraints (see `docs/`)

Detection + ByteTrack + ANPR/face are demo-live; multi-camera Re-ID and
criminal-DB face matching are roadmap. Weapon detection is now a **trained
`gun` object model** (`ibvap/weapon.py`, fine-tuned on YouTube-GDD) fused with the
2-D `AIM` posture heuristic — accuracy is bounded by that public dataset, so
treat it as demo-grade, not production. AGPL-3.0 applies (Ultralytics).
