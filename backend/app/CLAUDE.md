# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**IBVAP** — AI-Based Intelligent Video Analytics Platform for Border Surveillance
SIH 2026, PS 26187 · Ministry of Home Affairs / Sashastra Seema Bal (SSB) · Theme: Blockchain & Cybersecurity

Software-only AI analytics layer on top of existing CCTV infrastructure. No proprietary FRS/ANPR hardware. 1–2 week build window, team of 6 with coursework-level CV/ML experience.

---

## Commands

```bash
# Install PyTorch with CUDA 12.6 first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Mode A — Local OpenCV display** (webcam/file testing)
```bash
python main.py               # reads config.yaml
python main.py --verify-chain
```
Set `url: "0"` / `"1"` in `config.yaml` for laptop webcam, or a `.mp4` path.

**Mode B — Web server** (the main demo path)
```bash
python run_demo.py --cams 4        # server + 4 replayed CCTV streams + dashboard
python run_demo.py --cams 6 --src E:\my_footage
python run_demo.py --no-feed       # server only, for real phones
```

Or run the pieces separately:
```bash
python tools/make_test_videos.py            # build 4x 720p/24fps clips from the Kaggle dataset
python server.py                      # HTTP :8090 (monitor) + HTTPS :8443 (phones)
python scripts/tunnel.py                      # public URL for phones off the LAN (cloudflared/ngrok)
python scripts/feed_test.py --src test_videos --cams 8       # replay 8 streams (phone stand-in)
python scripts/feed_test.py --static --cams 8                # frozen frames (no-motion gate test)
python scripts/loadtest_mobile.py --cams 8 --seconds 40      # 8-device load test WITH measurements
python scripts/loadtest_mobile.py --static                   # no-motion test with skip-count report
python scripts/benchmark.py --cams 4          # four-way comparison of the throughput optimisations
python scripts/diagnose.py                    # end-to-end pipeline health check (server must be up)
python scripts/screen_watch.py --pick        # DEMO: detect + skeleton on your screen, transparent overlay
python server.py --mode screen       # run the WHOLE pipeline on a screen grab (menu if no --mode)
python training/datasets_fetch.py --weapon --dry-run   # plan the YouTube-GDD download into E:/Projects
python training/train_weapon.py --dry-run     # assemble the gun dataset, print counts, no training
python training/train_weapon.py               # fine-tune models/weapon_detector.pt (~1-2 h, RTX 3050)
```

**Unit tests** — pure Python, no GPU / model / network. Run any file directly
(exit 1 on failure), or `pytest tests/` if it's installed:
```bash
python tests/test_posture.py       # PoseClassifier geometry — Rule 5 (weapon posture)
python tests/test_risk_engine.py   # zone × time × behaviour scoring, weapon/armed fusion tiers
python tests/test_weapon.py        # gun detector: person association, temporal vote, label remap
python tests/test_thermal.py       # night/thermal fine-tune: VOC->YOLO, class remap, LLVIP split
python tests/test_anpr.py          # plate-format coercion, per-camera store, multi-frame vote
python tests/test_motion_gate.py   # frame-diff gate: skip / motion / force_every / hold
python tests/test_evidence.py      # SHA-256 hash chain: append, verify, tamper detection
python tests/test_anpr.py          # Indian-plate character coercion, per-camera reading store
python tests/test_geofence.py      # virtual fences: point-in-polygon, tripwire direction, debounce
python tests/test_snapshots.py     # intrusion snapshots: per-cam folders, index.jsonl, cooldown, frame dedup
python tests/test_anpr_events.py   # vehicle arrival/idle state machine, AnprResultWriter write/finalize
python tests/test_vehicle_type.py  # vehicle-type classifier's coarse-COCO fallback path
python tests/test_learn.py         # harvest pseudo-label gate: track-stable, rate limit, label format
python tests/test_calibration.py   # per-camera calibration: EMA bounds, suppression-cell promotion
```

- **Monitor** → `http://localhost:8090/monitor` (plain HTTP — no certificate warning)
- **Phone on the same WiFi** → `https://<LAN-IP>:8443/camera/0` (`getUserMedia`
  needs HTTPS; accept the self-signed cert once). Run `python tools/gen_cert.py` first.
- **Phone on any other network** (mobile data, different WiFi) →
  `python scripts/tunnel.py`, then open the `…/camera/0` URL it prints. Real cert, no
  warning. `run_demo.py --tunnel` does this alongside the server.

Two listeners share one app and one detection loop. Port 8080 is deliberately
avoided — Steam's webhelper squats on it and silently answers requests instead
of the server.

## CCTV / RTSP ingestion (`ibvap/rtsp_capture.py`)

`server.py` opens every entry in `config.yaml` `streams:` on startup, **in the
same process and pipeline as the phones**. A row with a real `url`
(`rtsp://…`, `http://…mjpg`, a webcam index, or a `.mp4` to loop) becomes an
`RtspCapture`; a **YouTube** watch URL becomes a `YouTubeCapture`
(`ibvap/youtube_capture.py`, see below); a row with `url: ws` (or no url) stays
a WebSocket slot a phone connects to. All of them land in the `captures` dict
and flow through the batched muxer → detector → dashboard identically — an NVR
channel, a YouTube link and a phone are indistinguishable downstream.

`RtspCapture` is built for real, flaky cameras: **one decode thread each** (a
frozen camera never stalls the muxer), **RTSP forced over TCP** with a 5 s
socket timeout, **`grab()` every loop but `retrieve()` (decode) only at
`cctv.decode_fps`** (15) to keep CPU cost low across 8–10 streams,
**auto-reconnect with backoff + a stall watchdog**, per-frame **resolution
normalise** to 1280×720, and **credential redaction** in every log line and in
`/status` / `/devices`. `POST /api/reconnect/<id>` force-cycles a frozen feed.
It is duck-compatible with `WebSocketCapture` (`.read()`, `.connected`,
`.info()`, `.latency_ms`, …) so nothing else changed.

Two flags on a capture decide how its frames are timed. **`paced`** holds a
source to its own frame rate (`_Pacer`): a recording demuxes far faster than it
plays — measured at 45× on a local clip — and an HLS live stream arrives a
segment at a time, so both need holding back or the picture races and the
behaviour engine, which measures speed per wall-clock second, reads a walk as a
sprint. RTSP and webcams are paced by their own socket and are left alone.
**`loop_at_end`** rewinds instead of reconnecting, for a recording only.

Full operator guide — how college CCTV is wired, vendor RTSP URL tables,
main vs sub-stream, what to ask IT for, `tools/rtsp_probe.py` /
`tools/discover_cameras.py` — is **`docs/CCTV_INTEGRATION.md`**.

## YouTube sources (`ibvap/youtube_capture.py`)

A watch URL in a stream's `url` becomes an ordinary camera — a dashboard tile
with boxes drawn on it, through detection, ANPR, face, fences and behaviour
like any other. `YouTubeCapture` subclasses `RtspCapture` and only overrides
**what to open**: `yt-dlp` resolves the link, and `_resolve_source()` (the hook
in the parent, called fresh on every reopen) hands back the media URL.

Three things that are not obvious. YouTube serves **no muxed formats** — every
stream is video-only, either a direct `https` MP4 or HLS, so yt-dlp's `best`
selector matches nothing and `pick_format()` chooses instead (H.264 first, the
tallest within `youtube.max_height`, a direct URL for a recording so it can
rewind, HLS for live). **DASH segment manifests are rejected** — OpenCV cannot
open one. And the media URL **expires after ~6 h and is bound to this machine's
IP**, so it is re-resolved before the deadline and dropped after any failed
open; retrying a stale link fails forever.

This is the only source kind that needs the internet — outbound only, no
inbound port. `yt-dlp` is imported lazily, so without it this one camera
reports a readable error and nothing else is affected; it also goes stale as
YouTube changes, so update it first if every link starts failing.

## Mobile ingestion (phones as demo cameras)

Phones stream into that same pipeline — no app install.

**How a phone connects.** It opens `https://<lan-ip>:8443/camera/<id>` (or
`/cam/<id>`) in its browser. `static/camera.html` calls `getUserMedia`, reads
what the phone *actually* granted from `track.getSettings()`, sends that as a
JSON `hello` over a WebSocket, then streams frames on the same socket.
`ibvap/ws_capture.py` decodes each frame and **normalises it to 1280×720
server-side** (phones rarely honour the request).

**Latency control — the phone is the bottleneck, not the pipeline.** A mobile
uplink is ~1–5 Mbps; 720p JPEG at 24 fps is ~10 Mbps, so with a naive sender
the WebSocket send buffer grows without bound and every frame the server sees
is *seconds* old (the "feed is 4–5 s behind" bug). `camera.html` fixes this by:
downscaling to a ≤960 px send canvas before encoding; capping the send rate;
and — the key part — **skipping the capture whenever `ws.bufferedAmount` still
holds the previous frame**, so latency stays flat instead of growing. When it
stays congested it drops to 10 fps / q0.40 and recovers automatically. Measured
localhost capture→receive is 2–6 ms; the server pipeline adds ~120–180 ms
(muxer tick + worker + mosaic).

**Latency indicator.** Each frame carries an 8-byte little-endian capture-ms
header; the `hello` carries `t0` for a coarse clock-sync (`serverTime` echoed
back in a `synced` reply). `ws_capture` reports a smoothed
`latency_ms` per camera in `/status` and `/devices`. The worker paints a badge
in the **top-right of each stream tile** — green `NET 120 ms`, or red
`NET 3.2s DELAYED` at ≥ 1.5 s — and the dashboard device panel shows the same.
`feed_test.py --latency-sim <ms>` backdates timestamps to exercise it without a
real slow link.

**Why WebSocket-JPEG, not MediaMTX/WHIP/WebRTC.** The brief recommended
`getUserMedia → WHIP → MediaMTX → RTSP → pipeline`. We use a direct
browser→WebSocket→pipeline push instead. Trade-off: WHIP/WebRTC gives H.264
hardware encode on the phone (less phone CPU, ~half the bandwidth) and ~150 ms
lower latency, and yields a standard RTSP URL. But it adds a separate MediaMTX
process to supervise, WebRTC ICE negotiation that some campus/corporate WiFi
blocks, and it would force the pipeline to *pull* RTSP with a decode thread per
stream — re-introducing exactly the per-stream cost the batched muxer is built
to avoid. The WebSocket path is one moving part, works through the cert already
set up, and was measured carrying 8×720p@24 with the GPU ~32% busy. If phone CPU
or bandwidth becomes the limit, MediaMTX is the upgrade path.

**Two ways in, pick by where the phone is:**

* **Same WiFi** → self-signed HTTPS on `:8443` (`tools/gen_cert.py`, SANs for every
  local IP). Fully offline, one command — the demo LAN may have no internet.
  Cost: one "Advanced → Proceed" tap per phone. If the laptop's IP changes
  (new WiFi), re-run `tools/gen_cert.py`.

* **Any network — mobile data, a different WiFi** → `python scripts/tunnel.py`. It
  runs `cloudflared` (no signup) or `ngrok` (one free `authtoken` — the script
  prints the exact setup steps if it's not configured) against the **HTTP**
  port `:8090`. The tunnel terminates TLS with a real, publicly-trusted cert,
  so the phone opens a normal `https://…/camera/0` with **no certificate
  warning at all**, and the WebSocket upgrades to `wss://` through it
  unchanged. The script prints the public `/monitor` + `/camera/N` links and a
  scannable QR. `run_demo.py --tunnel` launches it alongside the server; a
  failed tunnel never takes the LAN demo down.

  **The link is stable, not per-run.** ngrok's free plan gives every account
  one permanent static domain (e.g. `adjective-adjective-noun.ngrok-free.dev`)
  and plain `ngrok http 8090` binds to it — the URL only changes if the
  authtoken changes. `tunnel.py --domain <name>` (or `$IBVAP_TUNNEL_DOMAIN`)
  only renames it to something memorable, which needs a one-time free claim at
  dashboard.ngrok.com/domains. Buying a domain is never required for a fixed
  link; cloudflared's *free* quick tunnels, by contrast, are always random.

* **Your own domain** (e.g. `stream.mathswithsd.in/cam/0`) → a **cloudflared
  named tunnel**. `python scripts/tunnel.py --named <tunnel> --hostname <host>` (or set
  `$IBVAP_CF_TUNNEL` / `$IBVAP_CF_HOSTNAME` and use `run_demo.py --tunnel`).
  One-time setup — move the zone to Cloudflare DNS (the Netlify/other site
  keeps working with the same records), `cloudflared tunnel create/route`, fill
  `cloudflared.example.yml` — is written out in **`docs/CUSTOM_DOMAIN.md`**.
  `server.py` serves both `/cam/N` and `/camera/N`.

  `server.py`'s uvicorn runs with `proxy_headers=True` /
  `forwarded_allow_ips="*"` so it trusts the tunnel's `X-Forwarded-Proto/-For`.
  **The tunnel link is public and unauthenticated** — anyone with it can view
  the dashboard or push camera frames. Demo only; Ctrl+C to close it.

### Two-thread pipeline (muxer + inference worker)

`server.py` splits the work across two threads so a slow GPU pass can never
backpressure the camera sockets:

* **Muxer thread** (`_muxer_loop`) — sleeps to a fixed 24 Hz tick, drains each
  camera's queue into a one-deep latest-frame slot, snapshots all live slots,
  and hands the set to a depth-1 queue (`_infer_q`). It runs no model, no
  ByteTrack, no annotation, so it holds 24 tps regardless of inference cost.
  A busy-poll here (`sleep(0.001)` loop) was starving the other threads — it
  now sleeps the whole slack interval; the accumulator self-corrects jitter.
* **Inference worker** (`_inference_worker`) — owns the `Detector` and all
  per-camera ByteTrack / motion-gate state (single-threaded, no locks), plus
  risk scoring and evidence. Pulls the freshest snapshot and runs the full
  `process_batch` at its own pace. If it falls behind, the muxer **drops** the
  stale snapshot rather than growing a backlog — video and detection degrade
  gracefully to a lower rate while ingest stays real-time.

The `PIPE` log line reports both: `mux <tps>` (should stay 24) and
`worker <fps> (<bps>)` with a `dropped` counter.

**Everything in `process_batch` is batched across cameras.** `_infer_batch`
runs the detector once for the whole batch (`_track` then does per-camera
ByteTrack bookkeeping only), then `_pose_pass` and `_anpr_pass` each make **one**
forward pass over every person / vehicle crop in the batch — not one call per
camera. Per-track state (`PoseClassifier` aim streak / nose history,
`AnprEngine._plates`) is keyed by **`(cam_id, track_id)`** so track id 1 on two
cameras stays two independent tracks; a camera that was motion-gated this pass
keeps its state instead of having it flushed by another camera's pass.
There is **no server-push channel** — the dashboard polls `/status` at 2 Hz.

### Measured — 8 mobile streams, RTX 3050 Laptop 6 GB, YOLO26n TensorRT FP16

| metric | value |
|---|---|
| Aggregate ingest | **192.0 fps** (8 × 24, 0 drops) — real-time target MET |
| Per-stream ingest | 24.0 fps sustained |
| Muxer tick rate | 24.0 tps (never blocked by inference) |
| Inference cost | ~8–24 ms/frame (TensorRT engine, batch ~2) |
| Frames the GPU never saw | ~78–86% (rate cap + motion gate) |
| GPU utilisation | ~32% mean, 45% peak (`nvidia-smi` 2 Hz under load) |
| VRAM | 1.0 GB of 6 GB |
| Power | ~26 W · Temp ~64 °C |

Also verified: **6 moving + 2 static** → 191 fps, ~1 drop/sec; the two-thread
split is what let the fixed-shape engine hit real-time — on the old single
thread it managed ~160 fps. `.pt` FP16 on the split lands ~176–192 fps (a bit
client-starved in the load test); the engine is the better choice now.

**No-motion check** (`python scripts/loadtest_mobile.py --static`): 8 frozen streams
sent 192 fps; YOLO ran **~1 fps per stream** — only the deliberate `force_every`
keep-alive sweep (so a target already in frame at startup, or one creeping
slower than the pixel threshold, is still caught). ~96% of frames were
motion-gated away. It is not literally zero by design; raise `motion.force_every`
if you want it lower, but that trades away the safety sweep.

**Dashboard — connected devices.** The right rail lists every device: label,
CAM id, link state (live / idle / offline), negotiated resolution@fps, delivered
fps, and motion-gate skip %. **live** = motion gating is currently passing that
stream's frames to YOLO; **idle** = connected but static, no GPU spent on it.
The video is a single server-composited MJPEG **mosaic** (`/stream`), so N tiles
cost one HTTP connection — browsers cap at ~6 per host, which stalled tiles at
8 individual `/stream/<id>` connections. The layout bar over the mosaic switches
it between **Grid** (uniform) and **Focus** (1–4 chosen cameras large + the rest
in a right-side filmstrip) — `POST /api/layout`, composed server-side by
`_mosaic_tiles()`; `/status.mosaic.tiles` carries the rect map the fence-draw
tool clicks against.

## Code Structure

```
server.py             # FastAPI: WS + RTSP intake, muxer + inference-worker threads, MJPEG mosaic, /status.
                      #   --mode screen|phone|cctv|all  --surface dashboard|overlay
main.py               # OpenCV multi-stream grid (no web server). --verify-chain checks the hash chain.
run_demo.py           # One command: server + replay + browser  (--tunnel adds a public link)
config.yaml           # streams, model, throughput, risk, pose/weapon/geofence/learning thresholds

ibvap/                # the pipeline package (was src/; import as `from ibvap.X import …`)
├── detector.py       # Detector: batched YOLO26n + per-camera ByteTrack + carry-forward + pose/anpr/weapon passes
├── rtsp_capture.py   # RtspCapture: CCTV/RTSP/NVR/webcam/file puller — decode thread, TCP, reconnect + stall watchdog
├── ws_capture.py     # WebSocketCapture: browser/replay frames; generation-guarded reconnect
├── screen_capture.py # ScreenCapture: mss screen grab with the StreamCapture interface (server.py --mode screen)
├── motion_gate.py    # MotionGate: cheap frame-difference pre-filter in front of the GPU
├── posture.py        # PoseEstimator (yolo26n-pose on person crops) + PoseClassifier geometry rules
├── weapon.py         # WeaponDetector: trained `gun` model, held-gun association + temporal vote, fused in risk_engine
├── anpr.py           # AnprEngine: YOLOv8 plate detector on vehicle crops + threaded EasyOCR
├── geofence.py       # GeoFenceEngine: operator-drawn polygon zones / tripwire lines, image-space breach detection
├── snapshots.py      # SnapshotWriter: JPEG of the moment on a breach / weapon / crouch-lying, per-camera folders (docs/INTRUSION.md)
├── risk_engine.py    # RiskEngine: zone × time_of_day × behaviour → 0–100 score (+ weapon / fence overrides)
├── learn.py          # Harvester: auto-label stable/confident detections → data/learning/ pool (continuous learning)
├── calibration.py    # Calibrator: per-camera live confidence-floor + false-positive-region tuning, no training
├── evidence.py       # EvidenceChain: append-only SHA-256 hash-chain JSONL
├── event_store.py    # EventStore: SQLite store-and-forward (survives network outage)
└── display.py        # GridDisplay: OpenCV multi-stream grid with risk overlay (used by main.py)

scripts/              # operational CLIs
├── screen_watch.py   # transparent on-screen overlay: detector+pose+weapon+anpr (docs/SCREEN_WATCH.md).
│                     #   Also reached via `server.py --mode screen --surface overlay`.
├── feed_test.py      # Replays video / static frames as if they were phones (--static, --cams N)
├── loadtest_mobile.py# 8-device load test: phone-like WS clients, measures fps + GPU
├── benchmark.py      # Four-way comparison of the throughput optimisations
├── diagnose.py       # Checks each pipeline stage end to end (server must be up)
└── tunnel.py         # Off-LAN phones: ngrok / cloudflared quick tunnel, or --named for your own domain

training/             # model training + dataset prep  (weights → models/, datasets → E:/Projects)
├── datasets_fetch.py # --weapon (YouTube-GDD) / --thermal (LLVIP + optional FLIR ADAS) into E:/Projects
├── train_weapon.py   # Fine-tune yolo26n.pt → models/weapon_detector.pt (docs/WEAPON_DETECTION.md)
├── train_thermal.py  # Fine-tune yolo26n.pt for night/IR → models/registry.json "Thermal / night" (docs/THERMAL_NIGHT.md)
├── train_anpr.py     # ANPR: fine-tune the plate detector (→ models/license_plate_detector.pt) + prep the plate OCR (docs/ANPR.md)
└── retrain.py        # Continuous-learning fine-tune from the reviewed harvest pool (docs/CONTINUOUS_LEARNING.md)

tools/                # one-off setup / probes
├── gen_cert.py       # self-signed TLS cert for phone camera access on the LAN
├── make_test_videos.py # 720p/24fps demo clips with realistic static/motion mix
├── export_engine.py  # One-time TensorRT export (needs `pip install tensorrt`) — see note below
├── rtsp_probe.py     # Validate one camera/NVR URL: open time, res/fps/codec, jitter, verdict
├── discover_cameras.py # ONVIF WS-Discovery: list LAN cameras, print a paste-ready streams: block
├── nms_check.py      # verify YOLO26n is on the NMS-free inference path
└── test_detect.py    # quick single image/video/URL detection

notebooks/            # train_weapon_colab · train_thermal_colab · train_anpr_kaggle  (self-contained)
static/               # camera.html (phone page) · monitor.html (command dashboard)
docs/                 # architecture + per-feature guides (docs/README.md is the index)
models/               # license_plate_detector.pt · weapon_detector.pt · registry JSONs (fine-tunes .gitignored)
data/ · runs/         # auto-created runtime + training output (.gitignored)
cloudflared.example.yml # Named-tunnel ingress template (custom domain) — see docs/CUSTOM_DOMAIN.md
```

## Architecture

Two-track pipeline:

**Track A — Detection & Risk**
```
CCTV (RTSP/ONVIF) → Camera Tamper/Health Monitor + Edge Store-and-Forward
    → Video Ingestion & Frame Sampling (Day/Night/Thermal)
    → AI Analysis (YOLO26n + ByteTrack)
    → [Person | Vehicle (ANPR) | Threat Behaviour (posture) | Virtual Fence (ibvap/geofence.py — LIVE)]
    → Event & Risk Engine (zone × time × behaviour → 0–100 score; a breach of a Critical-severity fence — the default — forces Critical)
    → Command Dashboard (vanilla HTML: MJPEG mosaic + fence draw tool + /status poll)
```

Face-match (a small enrolled watchlist — `docs/FACE_RECOGNITION.md`), the fence engine and the dashboard are live; a map view is roadmap.

**Track B — Verification & Evidence**
```
Human Verify → Confirm → Intercept Dispatch
    → Event + Face Evidence (DB)
    → SHA-256 hash-chained ledger  (anchoring the chain on a blockchain: roadmap)
    → Secure Audit Log
```

False-alarm dismissals feed an Active Learning retrain loop back into the AI pipeline.

---

## Detection tuning (person / vehicle)

- **Class-restricted inference** — `model.classes: [0, 2, 3, 5, 7]` passed to
  `predict()` so YOLO only ever emits person + car/motorcycle/bus/truck. Cleaner
  output, a touch less NMS, no stray `chair`/`handbag` boxes.
- **Per-class confidence floors** — `predict(conf=…)` is set to the *lowest*
  bar, then `_track` filters `sv.Detections` **before the tracker**:
  keep a person at `model.person_conf` (0.30 — favour recall, people matter
  most), a vehicle at `model.vehicle_conf` (0.40 — higher bar kills
  parked-car / reflection false positives). The tracker never opens an ID on a
  weak or off-target box.
- **ByteTrack tuned for CCTV** (`tracker:` block) — `lost_track_buffer` 60
  detection frames (~7 s at 8 fps) so a person walking behind a pillar keeps
  their ID; `minimum_matching_threshold` 0.85. Falls back to library defaults
  on an older `supervision`.
- The **`medium` profile** (`yolo26m` / `yolo26m-pose`, 53.1 mAP) is one click
  away in the dashboard for when accuracy matters more than headroom.

## ANPR — number-plate recognition (`ibvap/anpr.py`)

Two-stage, ported from
[github.com/anindya-mukhopadhyay/ANPR](https://github.com/anindya-mukhopadhyay/ANPR)
(MIT): a YOLOv8 `license_plate` model (`models/license_plate_detector.pt`,
committed) runs on **vehicle crops** — event-triggered (see below), not
continuously — then **fast-plate-ocr / EasyOCR** reads the plate on a
**background worker thread** so it never touches the real-time budget.
`region: IN` coerces the common `O/0 I/1 B/8 S/5` confusions into a valid
`SS RR L(L)(L) NNNN` and marks it verified. Output: a tag under the vehicle
box, `data/plates/plates.csv` + a crop image, a `{"type":"plate"}` record in
the hash-chained evidence log, a yellow `PLATE` row in the dashboard alert log,
and `/status.anpr`.

`anpr.enabled: true` by default; needs `pip install easyocr` (or
`fast-plate-ocr`). If both are missing or the weights are missing, `AnprEngine`
logs one warning and disables itself — the rest of the pipeline is unaffected.
Full guide + tuning: **`docs/ANPR.md`**.

### Event-triggered ANPR + vehicle-type (`ibvap/anpr_events.py`, `ibvap/vehicle_type.py`)

ANPR does not re-scan a parked vehicle on every qualifying tick. `Detector._vehicle_pass`
(the renamed, restructured `_anpr_pass`) runs each vehicle track through
`VehicleArrivalTracker.classify()`: the tick a track first appears is its
**arrival** (fires once — a snapshot, the plate submit, and a vehicle-type
classify); a bounded number of **follow-up** ticks (`anpr.arrival_followup_reads`
/ `_s`) keep feeding `AnprEngine`'s multi-frame plate vote; after that (or once
a confident plate exists) the track is **idle** — zero inference work submitted
for it until it disappears and a genuinely different track_id arrives. A
camera with no tracked vehicles this pass is skipped entirely.

The **arrival snapshot** uses the camera's true native resolution when the
capture layer has one buffered (`RtspCapture`/`WebSocketCapture.get_native_frame()`,
stashed pre-downscale — see `ibvap/rtsp_capture.py` / `ibvap/ws_capture.py`),
falling back to the 1280x720 working frame otherwise. `ibvap/vehicle_type.py`
classifies it into the 8-class taxonomy (`Car / Pickup truck / Truck / Jeep /
2-wheeler / Tanker / Van / Auto-rickshaw`) — until `models/vehicle_type_classifier.pt`
exists, every result is a coarse COCO fallback (`car`/`two_wheeler`/`van`/`truck`
only, `conf: 0.0`, `fallback: true`). Train one on **Google Colab**:
**`notebooks/train_vehicle_type_colab.ipynb`** — pulls 4 Kaggle datasets (a
general Indian-vehicle set + dedicated auto-rickshaw and tanker sets + a
supplementary set), sorts every image into the 8 classes by a keyword
heuristic, fine-tunes a YOLO classification head. Honestly documented gap:
`jeep`/`pickup_truck` have no dedicated source dataset and come out thin —
the notebook prints per-class counts before training so this is visible, not
silent. `training/train_vehicle_type.py` + `training/build_vehicle_type_dataset.py`
are the local-machine equivalent (the latter bootstraps candidate crops from
this deployment's own harvested footage instead of Kaggle).

Both the picture and the extracted values (plate + type + both confidences)
land in a **dedicated output folder**, separate from `data/snapshots/` and from
`data/plates/plates.csv`: `data/anpr_results/<cam_id>/` + `index.jsonl`
(`AnprResultWriter`, `ibvap/anpr_events.py`), sha256-bound into the hash chain
like every other alert (`anpr_results.log_to_evidence`). A vehicle whose plate
never resolves is still filed, with `plate: null`, after
`anpr_results.finalize_timeout_s`. Read-only routes: `GET /anpr_snap/<cam_id>/<name>`
(the JPEG), `GET /api/anpr/results` (index tail); the Flutter app's **ANPR**
panel and alert log surface it (`AlertKind.anpr`).
`tests/test_anpr_events.py` (arrival/idle state machine + writer, no GPU),
`tests/test_vehicle_type.py` (coarse-fallback classification, no GPU).

## Posture / behaviour analysis (`ibvap/posture.py`)

When `pose.enabled`, `yolo26n-pose.pt` runs on person **crops** (not full
frames) and `PoseClassifier` applies pure-Python geometry rules to the 17 COCO
keypoints. Output feeds `risk_engine._behaviour()` and draws a skeleton + label.
Rules: `LYING` (skeleton wider than tall), `CROUCH` (knees near hips /
compressed height), `SCAN` (nose-x oscillating), `ARMS-UP` (wrist above
shoulder), `AIM` (Rule 5, below).

**Crowds (10–15+ people per stream).** Every person crop from every camera in a
detection batch goes through the pose head together. Each crop is resized to a
fixed `pose.imgsz` square (256) and each chunk padded to a fixed bucket
(`pose.max_batch` 24) — a constant input-tensor shape, so cuDNN autotunes at
warm-up instead of on every call. (Feeding raw variable-aspect crops made every
pose pass re-autotune: ~3 s vs ~45 ms — a latent bug that only showed once pose
ran on a genuinely crowded frame.) `pose.max_persons` (24) caps crops per camera
per pass; beyond it the biggest/nearest boxes win and the rest keep their last
skeleton — the detector and ByteTrack still count and follow everyone.
`model.max_det` (300) is the only cap on people detected and is far above this.
Measured, RTX 3050, ~15 people/stream all clearing the motion gate on one tick:
**2 streams ≈ 115 ms/detection-pass (within the 8 fps / 125 ms budget); 4
streams ≈ 190 ms** — over budget, so the effective detect rate dips to ~5 fps
during the surge while ByteTrack coasts the boxes (the two-thread design's
graceful-degrade path). If several cameras will *routinely* be that crowded,
drop `pose.imgsz` to 192, `detect_fps` to 6, or `pose.max_persons` to ~12.

**Rule 5 `chest_aim` / "AIM" — a weapon-READY POSTURE heuristic, not gun
detection.** It has no view of any weapon; it only asks whether the wrists and
elbows form a held two-handed grip. Clauses (all must hold, then persist
`aim_hold_frames` = **2** detection frames): both wrists between ~forehead and
~navel height, level with each other, within ~0.85 shoulder-width of each
other, in front of the torso (rejects folded arms — a folded wrist sits out
past the far shoulder), off the thighs, and at least one elbow no lower than
mid-torso (arm bent/forward, not hanging). This band is deliberately wide so a
real hold — high-ready, aiming across camera, low-ready muzzle-down — fires;
the earlier version was tight enough that a genuine toy-gun hold never
triggered. `tests/test_posture.py` (12 cases) checks the fire cases
(chest / low-ready / angled) and the reject cases (belt-clasp, folded arms,
one-hand, wide hands, surrender).

**A detected AIM forces the risk level to Critical** (`risk_engine.assess`:
`level = "Critical"`, `score = max(score, 92)`, `RiskAssessment.weapon = True`).
Previously a confirmed aim scored ~67 in daytime — under the 70 threshold — so
it only went Critical at night. The worker also paints a full-width red banner
across that tile (text depends on the fusion tier — see Weapon detection).

Known limits of a 2D-skeleton approach: a one-handed pistol grip won't fire
(needs two hands), and a person holding a clipboard/phone two-handed at chest
height is geometrically an aim. The object-level fix is now partly built — see
the next section.

## Weapon detection (`ibvap/weapon.py`, `training/train_weapon.py`)

The "custom-trained weapon object model" the posture section used to defer to is
now real (demo-grade). `models/weapon_detector.pt` is `yolo26n.pt` fine-tuned to
one class `gun` by `training/train_weapon.py` on
[YouTube-GDD](https://github.com/UCAS-GYX/YouTube-GDD) (5000 held/aimed/fired-gun
images). `WeaponDetector` runs **one extra batched `predict()` per detection
batch** — the same shape as `_pose_pass` / `_anpr_pass`, threaded through
`from_profile` / `from_weights` like `anpr=`. It never touches the
person/vehicle head.

**Calibration — a raw gun box is not trusted alone.** It is `confirmed` only
when: `conf >= weapon.min_conf` (0.45), area `>= weapon.min_box_area` (400 px²),
it is *held* (IoU with a tracked person box `>= weapon.person_iou_min`, or its
centre inside that box grown 15 %), **and** the same person track has carried a
held gun for `weapon.hold_frames` (3) consecutive weapon passes (temporal vote,
same idea as `aim_hold_frames`). Unconfirmed hits are drawn faint (`gun?`) and
never alert.

**Fusion in `risk_engine.assess`** (`RiskAssessment` gains `armed`,
`weapon_tier`), on the same track: AIM only → score ≥ 92, banner
`WEAPON (posture)`; confirmed gun only → ≥ 94, `GUN DETECTED`; AIM **and**
confirmed gun → **99, `ARMED THREAT`**. Each first-time `confirmed` gun writes a
`{"type":"weapon"}` row to the hash chain (same override shape as a fence
breach); crops go to `Harvester.submit_weapon` for operator review.

If `weapon.enabled` is true but `models/weapon_detector.pt` is absent,
`WeaponDetector` self-disables with a warning and the pipeline is unaffected
(the ANPR pattern) — so this ships safely before the model is trained.
`tests/test_weapon.py` (9 cases, no GPU) covers association, the vote, the
label remap. Full guide: **`docs/WEAPON_DETECTION.md`**. Datasets are downloaded
into `E:/Projects/ibvap-datasets/` (`weapon.dataset_root`), never the repo.

## Night / thermal model (`training/train_thermal.py`)

The Track-A pitch's "Day/Night/Thermal" is now partly real. `training/train_thermal.py`
fine-tunes `yolo26n.pt` on infrared imagery — **LLVIP** (fixed night
surveillance IR cams, person; `gdown`, no key) + optional **FLIR ADAS**
(thermal vehicles; needs a free Roboflow key or a Kaggle token). Source classes
are remapped to COCO ids and `data.yaml` is `nc: 80`, exactly like `training/retrain.py`,
so the fine-tuned head is drop-in for `Detector._classes = [0,2,3,5,7]`.

`register()` appends to `models/registry.json` with `label: "Thermal / night
<ts>"`, `kind: "thermal"`, and `base_mAP` / `ft_mAP` / `delta` in the same schema
as `retrain.register` — so it appears in the dashboard **MODEL** row with a
`+X.X mAP` badge and hot-swaps through the **existing** path
(`_read_registry` → `/api/models` → `/api/switch-model {weights_ref}` →
`_switch_q` dict → `Detector.from_weights`). **No `server.py` / `ibvap/detector.py`
/ `monitor.html` changes.** Operator workflow: swap to *Thermal / night* for IR
cameras, back to *Nano* for day. `tests/test_thermal.py` (6 cases, no GPU)
covers VOC→YOLO, the class remap, the LLVIP `19*`→val split. Full guide:
**`docs/THERMAL_NIGHT.md`**. LLVIP is non-commercial research licence — noted in
the docs.

## Input-source + render menu (`server.py --mode` / `--surface`)

`python server.py` prompts when run interactively (5 choices → `(input_mode,
surface)` in `_MENU_CHOICE`):
- `[1] screen → dashboard` — grab the screen as CAM-00 via `ibvap/screen_capture.py`;
  the whole server pipeline (detector, pose, weapon, risk, geofence, evidence)
  runs on it and shows at `/monitor`.
- `[2] screen → overlay` — `server.py` **does not start**; it `subprocess.call`s
  `scripts/screen_watch.py`, which paints transparent boxes / skeletons / **red gun
  boxes + a `GUN DETECTED` / `ARMED THREAT` banner** directly on the screen.
- `[3] phone`, `[4] cctv`, `[5] all` (every `config.yaml` `streams:` entry — the
  historical behaviour), all → dashboard.

`--mode screen|phone|cctv|all` + `--surface dashboard|overlay` skip the prompt; a
non-TTY stdin falls back to `("all","dashboard")`, and `run_demo.py` passes
`--mode all`. `_open_configured_streams()` branches on the module global
`_STARTUP_MODE`; the overlay dispatch happens in `__main__` before the server
starts. `scripts/screen_watch.py` runs the full analysis stack — detector + pose +
`ibvap/weapon.py` + `ibvap/anpr.py` (`--no-weapon` / `--no-anpr` / `--no-pose` to
drop any of them).

Per-track classifier state (nose history, height baseline, aim streak) is
flushed in `detector.py` for any ByteTrack id that drops out, so the dicts
don't grow and a stale streak can't carry into a reused id.

---

## Virtual fences / digital fencing (`ibvap/geofence.py`)

The operator draws **polygon zones** and **directional fence lines** (2+ points,
each segment a tripwire) on each camera's own view in the dashboard FENCES
panel. A tracked person/vehicle whose
**ground point** (bbox bottom-centre) enters a zone, or whose path crosses a
line in the flagged direction, forces that camera to **CRITICAL** —
`ra.level, ra.score = "Critical", max(ra.score, 90)` in `server._inference_worker`,
the same override shape as the weapon rule — and writes a `{"type":"breach"}`
record to the hash chain plus a `level="Breach"` row in `data/events.db`.

- **No geo-projection.** IBVAP has zero camera calibration (lat/lon, height,
  bearing, FOV, homography — none anywhere), so fences are **normalised 0..1
  image geometry per `cam_id`**, stored in `config.yaml` `geofence.store`
  (`data/fences.json`), edited live via `GET/POST/DELETE /api/fences` and handed
  to the worker over `_fences_q` (the `_switch_q` pattern). A camera-marker map
  is spec'd in the plan but deliberately unbuilt — it would be decoration, not
  the digital-twin projection the "GIS" slogan implies.
- `GeoFenceEngine.evaluate(sr)` is edge-triggered (only *new* breaches); polygon
  exit is debounced by `geofence.exit_passes`, tripwire re-fire by
  `geofence.reentry_cooldown_passes`. Per-track state keyed `(cam_id, track_id)`;
  `prune()` drops stale ids like `PoseClassifier.flush_missing`.
- Pure geometry, no GPU/model/OpenCV — `tests/test_geofence.py` (16 cases).
- Overlay is burned into the mosaic (`server._draw_fences`) **and** drawn as an
  SVG layer in `monitor.html`; the click→tile→cam_id math undoes the mosaic's
  `object-fit: contain` using `/status.mosaic.tiles` (an explicit `{cam_id,x,y,w,h}`
  rect map) so it matches `_build_mosaic` in both **grid** and **focus** layouts.
- A breach also saves a **snapshot of the moment** — see below.
- Full reference: **`docs/GEOFENCE.md`**.

---

## Intrusion snapshots (`ibvap/snapshots.py`)

On a **fence breach**, a **confirmed weapon**, or a **malicious posture**
(crouching / lying — `snapshots.malicious_postures`), `_inference_worker` writes
an annotated JPEG *and* a raw JPEG of that frame under
`data/snapshots/<cam_id>/` (folder `0` for camera 0, `1` for camera 1 …, created
the first time each camera produces a frame). Each event appends one line to that
folder's `index.jsonl` and the annotated image's SHA-256 is embedded in the
breach / weapon / `{"type":"posture"}` hash-chain record, so the file is
tamper-evident. `SnapshotWriter.should_capture()` debounces per
`(cam_id, track_id, reason)` by `snapshots.cooldown_s`; several intruders
crossing in one detection pass share one JPEG (`deduped` flag). Posture is logged
at `"Posture"` severity and does **not** force Critical unless
`snapshots.escalate_posture`. Encoded inline on the worker (rare, edge-triggered).
Read-only routes: `GET /snap/<cam_id>/<name>` (the JPEG), `GET /api/snapshots`
(index tail); the dashboard ALERT LOG shows the thumbnails.
`tests/test_snapshots.py` (11 cases, no GPU). The operator can also switch the
mosaic between **Grid** and **Focus** (1–4 mains + right-side filmstrip) from the
layout bar — `POST /api/layout`, `_mosaic_tiles()`. Full reference:
**`docs/INTRUSION.md`**.

---

## Continuous learning (`ibvap/learn.py`, `ibvap/calibration.py`, `training/retrain.py`)

The pitch docs' "Active Learning retrain loop" (`CLAUDE.md` Track B) is now
partly real. **No weights ever change on their own.**

- **Harvest (always on).** `Harvester` is hooked at the top of the
  `_inference_worker` `for sr in results` loop and into `screen_watch.py --learn`.
  A detection is auto-labelled into `data/learning/` only when it is **confident**
  (`learning.harvest.min_conf`) **and track-stable** (same `(cam,tid)` for
  `min_track_frames` inferred passes); a frame with any box in the ambiguous band
  is skipped. Rate-limited + deduped + `max_pool`-capped. The `submit()` on the
  worker thread is cheap (dict lookups + one gated `frame.copy()`); a daemon
  thread does the JPEG encode + `manifest.jsonl` append (like `AnprEngine`'s OCR
  worker). Classes are **not** remapped — labels use COCO ids `0,2,3,5,7` and
  `training/retrain.py` writes `data.yaml` `nc: 80`, so the fine-tuned head is drop-in.
- **Calibration (always on, no training).** `Calibrator` is owned by the worker
  and threaded through hot-swaps like `anpr=`. `observe()` (in `Detector._track`,
  inferred passes) tracks which detections form stable tracks vs die young; it
  rolls a per-camera `person_delta`/`vehicle_delta` (bounded
  `[-headroom, +max_tighten]`) and promotes repeatedly-noisy `32x18` grid cells
  to a suppression mask. Applied in the pre-tracker filter in `_track` — the
  batched `predict()` is untouched, and `_predict_conf` is pre-lowered by
  `headroom` so a negative delta can actually recover boxes. Nothing applies
  below `min_obs_before_apply`; state persists to `data/calibration.json`.
- **Fine-tune + hot-swap (deliberate).** `training/retrain.py` (CLI or the dashboard
  **Retrain** button → `subprocess.Popen`, stdout tailed into
  `/status.learning.last_train`): assemble the `keep`/`background` items into
  `datasets/ibvap_live/` (chronological split), `YOLO("yolo26n.pt").train(freeze=10)`
  + a COCO-val replay slice, `model.val(classes=[0,2,3,5,7])` for base vs
  fine-tuned mAP, register `models/yolo26n_ft_<ts>.pt` in `models/registry.json`.
  `_switch_q` now carries a **weights dict** as well as a profile name;
  `Detector.from_weights` (path-allowlisted to `models/`) builds it; `/api/models`
  merges the registry so the fine-tuned model appears in the MODEL row with a
  `+X.X mAP` badge. Rollback = switch to `nano`.
- Endpoints: `GET /api/learn/pool`, `GET /learn/thumb/{id}.jpg` (hand-rolled
  bytes route), `POST /api/learn/review` (mirrors `/api/fences`),
  `POST /api/learn/calibration`, `POST /api/learn/retrain`.
- Full reference: **`docs/CONTINUOUS_LEARNING.md`**. `tests/test_learn.py` +
  `tests/test_calibration.py` (no GPU).

---

## Multi-stream throughput — how the pipeline stays real-time

Five mechanisms, in the order a frame meets them. `scripts/benchmark.py` measures 1–4
in isolation on the dev RTX 3050; the live 8-stream numbers are under
"Two-thread pipeline" earlier.

**1. Fixed-tick muxer + inference worker** (`server.py`)
`server.py` runs two threads. The **muxer** sleeps to a 24 Hz tick, drains every
camera's queue into a one-deep slot, and hands the whole set of live slots to a
depth-1 queue. The **inference worker** pulls the freshest set and runs the
model + tracking + risk at its own pace; if it falls behind, the muxer drops
the stale snapshot. Because every camera contributes to every tick, the batch
is always "all connected cameras" (nvstreammux's job) — and because inference
is off the muxer thread, a slow GPU pass never backpressures the sockets.

**2. Detection-rate cap on a global tick** (`detect_fps` = 8, `stream_fps` = 24)
The muxer ticks at 24 Hz. Every 3rd tick a fresh YOLO + ByteTrack pass runs
(8 fps); on the two ticks between, each box is advanced by its last measured
per-tick velocity, so the annotated stream still updates at the full 24 fps —
smooth boxes, one third of the GPU cost. The schedule uses one shared tick
rather than each camera's own frame counter, so all cameras come due in the
same pass and the batch stays full. (This is velocity extrapolation, not a
Kalman predict — fine for people at demo range; a real 24 fps Kalman step is a
possible future refinement.)

**3. Motion gating** (`motion_gating`, `ibvap/motion_gate.py`)
A 160×90 greyscale frame difference (0.93 ms) decides whether a scene changed
at all. Static footage never reaches the GPU. A gated camera advances its
schedule as though it had run, so it stays in phase with the others. Two
safeguards prevent a missed intrusion: `force_every` sweeps every camera at
least once a second regardless, and `hold_frames` keeps the gate open briefly
after motion stops.

**4. Fixed batch shapes** (`_bucket` in `ibvap/detector.py`)
**This one is not optional and is easy to regress.** Ultralytics/cuDNN re-tune
kernels whenever the batch size changes (a run of varying batches averaged
**304 ms/pass** vs **46 ms** for a constant batch of 12 — 6× slower), and a
TensorRT engine simply *cannot* take a shape other than the one it was built
for. So batches are padded up to a fixed ladder — `[1, 2, 4, 8, 16]` for a
`.pt` model, or `[max_batch]` only for a `.engine` — and the padded outputs
discarded. The warm-up primes exactly these shapes.

**5. Motion-vs-forced live state** (`ibvap/motion_gate.py`)
The gate records whether its last pass was real motion or just the keep-alive
sweep, so the dashboard's live/idle state reflects genuine activity — a swept
static camera stays "idle".

### Measured results (4 × 720p @ 24 fps, RTX 3050 6 GB)

| configuration | aggregate fps | per camera | GPU frames | real-time |
|---|---|---|---|---|
| baseline (1 frame/call, every frame) | 75.6 | 18.9 | 1152 | no |
| + batching | 203.2 | 50.8 | 1152 | yes |
| + rate cap (detect 8 fps, track 24 fps) | 468.1 | 117.0 | 384 | yes |
| + motion gating | 466.6 | 116.7 | 213 | yes |

**6.2× faster, 82% fewer frames inferred.** 96 fps is needed to keep up; the
pipeline sustains ~467, leaving headroom for roughly 19 streams. At 12 streams
it holds 526 fps against the 288 fps needed.

**Read those as GPU-side headroom only.** The sources were phones and
pre-loaded video, so H.264 decode is not in the number. Real RTSP is decoded on
the CPU (no NVDEC), and `docs/CCTV_INTEGRATION.md` already notes 8–10 cameras
pegging it. **No real-camera count has been measured** — do not present "19
streams" as 19 CCTV cameras.

Benchmark numbers are only meaningful after a warm-up run — cuDNN autotuning
made the first configuration measured look 4× slower than it was, which briefly
made motion gating appear to be a regression. `scripts/benchmark.py` now discards a
warm-up pass; keep it that way.

---

## Locked Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Detection | **YOLO26n** (Ultralytics ≥8.3.0) | 40.9 mAP, 1.7 ms T4 TRT, 2.4M params |
| Tracker | **ByteTrack** | 80.3 MOTA; faster than DeepSORT, default in Ultralytics pipeline |
| ANPR | YOLOv8 plate detector + EasyOCR (`ibvap/anpr.py`) | Live. On vehicle crops only; threaded OCR. `docs/ANPR.md` |
| Face recognition | RetinaFace + ArcFace (InsightFace `buffalo_l`) | Live against a small enrolled watchlist (team photos), burst-voted. `docs/FACE_RECOGNITION.md`. Non-commercial-research weights |
| Re-ID | OSNet | Architecture/roadmap target; demo uses timestamp + visual heuristic |
| Backend | FastAPI + WebSocket + SQLite (WAL) | PostgreSQL/PostGIS is roadmap — not used |
| Frontend | Flutter Windows console + vanilla-HTML dashboard (`static/monitor.html`) | React/Tailwind/Leaflet is roadmap — not used |
| Blockchain | SHA-256 hash-chain ledger | Tamper-evident local log, not a distributed network. Anchoring the chain tip on a blockchain is roadmap |
| Edge buffer | SQLite store-and-forward queue | Drained by `ibvap/alert_forward.py` to webhook / syslog-CEF / MQTT sinks. Delivered events kept `alerts.retention_days` (default 7); undelivered events are kept until delivered |
| Deployment | Windows + NVIDIA CUDA workstation (Flutter console, bundled Python) | Docker and Jetson Orin are roadmap — no Dockerfile or aarch64 build exists, and an NVIDIA GPU is required |

**Python:** 3.11.9 (NOT 3.13/3.14 — PyTorch unsupported)
**CUDA:** 12.6 · **Inference precision:** FP16 for deployment, FP32 for dev
**Dev GPU:** RTX 3050 laptop, 6GB VRAM — this is the minimum hardware everything must run on

### TensorRT engine

`tensorrt-cu12 == 10.13.3.9` **is installed** (GPU-only library; the 1.5 GB
`tensorrt_cu12_libs` wheel is the CUDA-12 payload — matched to torch's cu126,
*not* the newest 11.x, which changed the builder API Ultralytics 8.4 expects).
`yolo26n.engine` **is built** — `tools/export_engine.py`, FP16, fixed `(8, 3, 640,
640)` shape, TensorRT 10.13, specific to this RTX 3050.

**It is the pipeline default:** `weights: "yolo26n.engine"` in config.yaml.
The two-thread split (above) is what makes this work — on the old single
thread the fixed-shape engine's per-tick 8-image pass dropped the muxer to
~160 fps; with inference on its own worker the muxer holds 24 tps and the
engine sustains 192.

| | .pt PyTorch FP16 | .engine TensorRT FP16 |
|---|---|---|
| detections | reference | **identical** (batch 1/2/4/8 all match) |
| batch-1 latency (isolated) | 27 ms | **5.6 ms — 4.8× faster** |
| batch-8 latency (isolated) | 37 ms | 28 ms |
| **8-stream, two-thread pipeline** | ~176–192 fps | **192 fps, 0 drops** |
| VRAM | 0.9 GB | 1.0 GB (fixed shape); 4.6 GB if built `dynamic=True` — rejected |
| warm-up | ~11.6 s (cuDNN autotune) | ~0.5–1.7 s |

The engine's input shape is fixed at `max_batch`, so the detector pads every
detection tick to a full `max_batch`-image pass — cheap once inference is off
the muxer thread. A `dynamic=True` engine avoids the pad but reserved ~4.6 GB
on the 6 GB card and stalled on shape changes — rejected.

**Fall back to `weights: "yolo26n.pt"`** only when the `.engine` is missing
(not exported on this machine yet) or on a non-CUDA box. `ibvap/detector.py`
detects `.engine`/`.onnx` (`self._is_engine`) and skips the PyTorch-only
`.to()` / `quantize=` calls automatically. Re-export per GPU (the engine is
hardware-specific): `python tools/export_engine.py` — reads `image_size` and
`max_batch` from config, builds FP16, ~5–10 min.

---

## Hard Numbers — Do Not Change Without Flagging

- **AI inference stream:** 720p (1280×720) @ 15–25 fps via RTSP/ONVIF substream — never the primary 4K recording stream
- **Effective inference rate:** 8–10 fps (`detect_fps: 8`), with ByteTrack carrying boxes at the full 24 fps
- **Measured capacity:** 4 × 720p/24fps at ~467 fps aggregate on an RTX 3050 6 GB — **GPU-side only** (phone / pre-loaded sources, decode excluded; ~19 streams of GPU headroom). No real-RTSP camera count has been measured; expect CPU decode to bind first
- **Risk score:** 0–100; zone sensitivity 40% + time-of-day 20% + behaviour 40%; threshold **≥50** = High. A score alone never reaches Critical (`risk.score_can_reach_critical: false`) — Critical is reserved for a confirmed weapon, a watchlist match, or a breach of a Critical-severity fence (the default)
- **YOLO26n COCO mAP:** 40.9 (50–95); latency 1.7 ms T4 TRT / 38.9 ms CPU ONNX
- **Evidence retention:** target 90 days local, then archived to central command — **not implemented**: snapshot, ANPR, face and evidence folders grow without limit (only the alert queue in `events.db` is pruned)
- **False-positive target:** <15% at launch — **a target, not a measurement**: no false-alarm rate has been measured for any detector yet
- **License:** AGPL-3.0 (all Ultralytics models) — acknowledge openly, do not hide

---

## Hard Constraints

- **Criminal face matching:** No live matching against any criminal database in the demo. Use consented mock watchlist (team photos or LFW). Frame any match as a lead for human verification. Real deployment = API call to NCRB CrPI.
- **Camera stream:** Always run inference on the secondary/AI substream (720p), never primary 4K. ~4–9x compute saving with no detection quality loss.
- **GIS terminology:** Call it a "GIS Digital Map," not "Digital Twin" — a twin implies live 3D sync that is not being built.
- **Virtual fences are image-space, not geo-projected.** No camera calibration (lat/lon, height, bearing, FOV, homography) exists anywhere in the platform, so a fence is a normalised polygon/line drawn on one camera's view — not a shape on a map. A camera-marker map is roadmap; do not imply fence footprints are projected onto ground coordinates.
- **Re-ID:** Do not claim OSNet multi-camera Re-ID is live. Present it as roadmap. Demo uses simplified heuristic.
- **TensorRT:** Only PyTorch, TorchScript, and TensorRT actually use the Jetson GPU — all other export formats are CPU-only.
- **YOLO + tracker naming must be consistent across all slides and all code.** The locked choices are YOLO26n + ByteTrack.

---

## Detection Range Tiers (state explicitly, not as a limitation)

| Tier | Camera | Range | Capability |
|---|---|---|---|
| Full analytics | Fixed bullet/dome 2–4MP | 0–80m | Detection + ANPR + face + behaviour |
| Detection only | Long-range PTZ 30–45x | 80–300m | Person/vehicle reliable; ANPR/face degrade past ~150m |
| Thermal presence | Bi-spectrum thermal | 1.5–8km | Long range: movement/presence only. Close/mid range: person + vehicle *classification* via the swappable **Thermal / night** model (`training/train_thermal.py`, LLVIP + FLIR ADAS) — see `docs/THERMAL_NIGHT.md`. |

62% of users attempting facial ID beyond 70 ft (21m) with a 4MP camera report failure — resolution and lens determine range, not AI.

---

## What to Present as Roadmap (Not Demo-Live)

- Multi-camera Re-ID (OSNet)
- Criminal DB face matching (→ NCRB CrPI API)
- VLM-based explainable alerts
- Feed-spoofing / replay-attack detection
- **Weapon detection — a trained `gun` model now exists** (`ibvap/weapon.py`,
  `training/train_weapon.py` on YouTube-GDD), fused with the `AIM` posture heuristic
  (`ibvap/posture.py` Rule 5). Present it as **demo-grade**: accuracy is bounded by
  a single public dataset, so misses on odd angles / small guns / low light are
  expected. Production accuracy (more data, operator-confirmed crops, a heavier
  backbone) is the roadmap item, not the capability itself.
- Per-endpoint auth (camera intake, dashboard, tunnel link are all currently open)
- **ANPR accuracy — now partially live.** `ibvap/anpr.py` has: a `fast_plate`
  OCR backend (`fast-plate-ocr`, plate-specific ONNX) with an EasyOCR fallback;
  **multi-frame positional voting** across each vehicle track; and a
  `training/train_anpr.py` (+ `notebooks/train_anpr_kaggle.ipynb`) that
  fine-tunes the YOLO plate detector on Indian datasets and the OCR on Indian
  crops (folding in operator-confirmed crops from `data/learning/plates/`).
  Present as: *demo-grade, tunable* — bounded by the public Indian-plate data;
  two-row plates and heavy motion blur are still hard.

---

## Project Documentation

All reference material lives in `docs/`:
- `IBVAP_Project_Context_TechStack_1.md` — full project context, architecture decisions, hardware cost math
- `IBVAP_Technical_Specifications_1.md` — submission-ready numbers with [HARD]/[TUNABLE]/[MISSING] tags
- `IBVAP_References.md` — 22 numbered citations (R1–R22) for every hard number

## Agent Workflows

`.agent/workflows/` contains named design-pass workflows (adapt, animate, audit, bolder, clarify, colorize, critique, delight, extract, harden, normalize, onboard, optimize, polish, quieter, simplify, teach-impeccable). `.agent/skills/frontend-design.md` governs all frontend UI work.
