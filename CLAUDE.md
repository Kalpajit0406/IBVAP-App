# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the **master briefing** for the whole product tree (`E:\IBVAP app`): what the project is for,
what exists, how it works, what is measured versus assumed, and what is left to do. It is written so a
new session can be productive without the conversation history. The backend has its own, deeper file —
[`backend/app/CLAUDE.md`](backend/app/CLAUDE.md) — covering the throughput engineering, per-feature
internals and measured benchmark tables; this file points into it rather than repeating it.

Last full refresh: **2026-09-21**, at commit `3e33b06`. If the repo has moved on, trust the code over
this file and fix the file.

---

## 1. What this project is, and the goal

**IBVAP — Intelligent Border Video Analytics Platform.**
Smart India Hackathon 2026 · problem statement **PS 26187** · Ministry of Home Affairs / Sashastra Seema
Bal (SSB) · theme **Blockchain & Cybersecurity** · team "Glitch Gods" (6 people, coursework-level CV/ML
experience, a short build window). Repo: `github.com/Kalpajit0406/IBVAP-App` (`main`).

### The problem being solved

Border forces run CCTV at Border Out Posts (BOPs), check posts, border roads and strategic sites. A
conventional CCTV system records and shows video and needs a human watching continuously. Facial
recognition (FRS), number-plate recognition (ANPR), intrusion detection and object tracking normally
need specialised hardware and proprietary software, which makes them costly to deploy at remote posts.

### The goal

A **software-only** AI platform that turns *existing* IP CCTV into an intelligent surveillance network —
no dedicated FRS/ANPR/smart-camera hardware — that ingests standard camera streams, analyses them in real
time, and tells the right people, at the right moment, what matters. It must be cost-effective, scalable,
suitable for remote deployment, and able to integrate with existing command and control (C2) systems.

### The problem statement's requirement list

The platform "should provide capabilities such as": human detection and tracking · vehicle detection and
classification · face detection · ANPR · virtual fence intrusion detection · suspicious activity detection ·
night-time movement detection · real-time alert generation and event logging.

The expected solution "should": eliminate dependence on expensive dedicated surveillance hardware · enable
intelligent monitoring through AI video analytics · provide real-time alerts for incidents and border
intrusions · support facial recognition, vehicle identification and behavioural analytics through software ·
improve situational awareness and response time · **support integration with existing command and control
systems** · be cost-effective, scalable, and suitable for remote border locations.

The statement was re-checked word-for-word on 2026-09-21 against the version first audited: **identical**.

### What the product physically is

A **standalone Windows product that runs on one machine with an NVIDIA GPU**. Two parts that ship together:

1. A **Flutter desktop console** (`lib/`, Windows only) — the operator UI. It can start and stop the backend.
2. A **Python backend** (`backend/app/`) — FastAPI + a GPU inference pipeline. Bundled inside the app; not a
   separate project.

Phones can act as demo cameras through a browser page (no app install). Real CCTV comes in over RTSP.
Nothing needs the cloud. Target/dev hardware: **RTX 3050 Laptop GPU, 6 GB VRAM** — the minimum everything
must run on. Python **3.11.9** (not 3.13/3.14: PyTorch), CUDA **12.6**, torch **2.13.0+cu126**.

---

## 2. Standing rules — read before changing anything

These come from the user, mostly as corrections. Each has a reason; keep the reason in mind when a case
falls between the rules.

1. **All work happens in `E:\IBVAP app`.** `E:\IBVAP` is the older backend source checkout, kept as
   *reference only* — read it, never edit it. The backend bundled at `E:\IBVAP app\backend\app` is part of
   the app and is in scope.
2. **Standalone and local.** The server and the app are one product on one machine. Default to loopback
   listeners and **outbound-only** external traffic. The only things that should leave the machine are
   outbound **alerts/signals** and anything **blockchain**-related. Treat any inbound exposure as something to
   justify, not a default. (Current config defaults do not meet this yet — see §11, item E.)
3. **Always target CUDA.** If `torch.cuda.is_available()` is False, fix the torch install
   (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`); never settle for a CPU
   fallback and report it as working.
4. **A detected person alone must never reach Critical.** People present — however many, at night, in a
   sensitive zone, with odd posture — tops out at **High**. Config: `risk.score_can_reach_critical: false`,
   `risk.posture_aim_level: "High"`. Critical is reserved for a confirmed weapon / armed person, a watchlist
   face match, or a breach of a Critical-severity fence. The behaviour rules (running, group, following)
   default to **Medium** and cannot reach Critical on their own. Keep this when touching the risk engine,
   severities, or the console's threat picture.
5. **Test detection on video, not stills.** Real deployment ingests several concurrent feeds; default to
   multi-stream video at realistic resolution and frame rate for demos, benchmarks and regression checks.
6. **Downloaded datasets go under `E:/Projects/ibvap-datasets/`**, never inside the repo. *Exception:* the
   face-watchlist enrollment photos ship with the product at
   `backend/app/models/face/gallery_photos/<identity>/` (needed to rebuild `gallery.json`); never move them out.
7. **Show logs during long computation** — periodic progress (frames, fps, inference ms, device, per-camera
   counts). Do not suppress uvicorn access logs or run quiet.
8. **Performance techniques the user specifically wants:** batched multi-camera inference (nvstreammux-style),
   motion-gated inference, and a reduced detector rate with full-rate tracking; scale to an arbitrary number of
   streams dynamically. These are built (see §6.4); do not regress them.
9. **Do not commit or push unless asked.** Commits so far were requested explicitly or made by the user
   (`b12cefe` and `3e33b06` were not made by an agent). A `git push` from an agent has been blocked before;
   the user runs `git push origin main` themselves.
10. **Claims must match what is built and measured.** This is graded work and the honesty is deliberate. Do
    **not** claim: "19 CCTV streams" (that was GPU headroom on phones/preloaded video, decode excluded);
    Docker or Jetson support (none exists); a "72-hour offline buffer" (it is `alerts.retention_days`, default
    7); blockchain (it is a local SHA-256 hash chain — anchoring is roadmap); live Re-ID; geo-projected fences;
    a "Digital Twin" (say "GIS Digital Map"); live criminal-database matching (demo uses a consented mock
    watchlist); a measured false-alarm rate (none exists).
11. **Naming is locked:** detector **YOLO26n**, tracker **ByteTrack**, everywhere — code, docs, slides.

---

## 3. Status at a glance

Legend: **Done** = built and exercised · **Partial** = built with a stated gap · **Missing** = not built.

| PS requirement | Status | Where / what is true today |
|---|---|---|
| Human detection & tracking | **Done** | YOLO26n (TensorRT FP16 engine) + ByteTrack, per-camera state, new tracks withheld until confirmed by a second overlapping match. `ibvap/detector.py` |
| Vehicle detection & classification | **Done** (classifier weights untracked) | COCO car/motorcycle/bus/truck + an 8-class Indian vehicle-type classifier. `vehicle_type_classifier.pt` exists locally but is gitignored, so a fresh clone falls back to coarse COCO classes |
| Face detection | **Done** | RetinaFace + ArcFace (InsightFace `buffalo_l`), always snapshots a face (blurry/sideways/low-light included), burst-voted watchlist match. `ibvap/face.py`, `face_events.py` |
| ANPR | **Done** (demo-grade) | Plate detector on vehicle crops + threaded fast-plate-ocr / EasyOCR, multi-frame vote, Indian plate format coercion. `ibvap/anpr.py`, `anpr_events.py` |
| Virtual fence intrusion | **Done** | Polygon zones + directional tripwires, loiter, crossing counts, per-fence severity, night arming windows, inbound label, close-following. **Image-space, not geo-projected.** `ibvap/geofence.py` |
| Suspicious activity | **Partial** | Weapon (gun model + AIM posture), crouch/lying, loiter, **running, group formation, close-following crossing**. Missing: abandoned object, person leaving a vehicle. No false-alarm rate measured for any of them |
| Night-time movement detection | **Partial — weakest item** | Motion gate works at any light, and a *manually* swapped thermal/night model exists (recall 0.53→0.72 on infrared LLVIP). **No automatic day/night switching, no low-light enhancement**, and the thermal weights are gitignored |
| Real-time alerts & event logging | **Done** | Event schema v2 (ULID ids, severity/category split), SQLite store-and-forward queue, forwarder to webhook / syslog-CEF / MQTT sinks with per-sink cursors, backoff and retention |
| Integration with C2 | **Partial** | Cursored `GET /api/events`, SSE stream, ack endpoint, CSV/JSON export, a documented wire format. Gaps: reads are unauthenticated by default, one token with no scopes, no OpenAPI response models, console acks never reach the server |
| Eliminate special hardware | **Partial** | Software-only on an NVIDIA GPU + standard RTSP. **Never tested against a real IP camera or NVR** — only phones, files, screen capture and YouTube |
| Cost-effective / scalable / remote | **Partial** | GPU headroom measured (8 phone streams, ~32% GPU). Real RTSP is CPU-decode-bound; the camera count is an estimate (~8–12), not measured |
| Blockchain (the *theme*) | **Partial** | SHA-256 hash-chained evidence ledger, tamper-evident and verifiable. **Not anchored anywhere** — see §11, item C |
| Command & control demo surface | **Partial** | `scripts/alert_sink.py` receiver proves alerts leave the box. No "Command Centre" page, no Telegram sink |

Everything not "Done" above is in the prioritised backlog in **§11**.

---

## 4. Repository map

```
E:\IBVAP app\                      <- the product (git: origin = Kalpajit0406/IBVAP-App, branch main)
├── CLAUDE.md                      <- this file
├── README.md                      <- Flutter console readme (STALE — see §11, item L)
├── SETUP_AND_OPERATIONS.md        <- operator guide, 13 sections (updated for YouTube; says "eight tabs", there are nine)
├── package.ps1                    <- builds the standalone release: flutter build + mirror backend/ next to the .exe
├── pubspec.yaml / pubspec.lock    <- Flutter deps: only `http` and `qr_flutter` (deliberately dependency-light)
├── analysis_options.yaml
├── .agent/                        <- design-pass workflows + frontend-design skill (UI polish conventions)
├── lib/                           <- Flutter Windows console (11.8k lines) — see §6.9
├── test/widget_test.dart          <- 40 Flutter tests
├── windows/                       <- Windows runner (the only platform folder)
└── backend/
    ├── README.md                  <- how the bundled Python runtime is built (and why it has its own Lib/DLLs/tcl)
    ├── runtime/                   <- (gitignored, several GB; DOES NOT EXIST in this checkout — see §11, item K)
    └── app/                       <- the IBVAP server. Run everything from here; paths are relative to it
        ├── server.py              <- FastAPI app, muxer + inference threads, MJPEG mosaic, all HTTP/WS routes (2.8k lines)
        ├── main.py                <- OpenCV-window mode without the web server; `--verify-chain`
        ├── run_demo.py            <- server + replayed CCTV streams + browser in one command
        ├── config.yaml            <- ALL tunables, heavily commented (start here to understand a knob)
        ├── requirements.txt
        ├── CLAUDE.md              <- backend deep-dive (throughput, per-feature internals, benchmarks)
        ├── ibvap/                 <- the pipeline package (27 modules, ~9k lines) — table below
        ├── tests/                 <- 23 pure-Python test modules, no GPU/model/network needed
        ├── scripts/               <- alert_sink.py, feed_test.py, loadtest_mobile.py, benchmark.py, diagnose.py, screen_watch.py, tunnel.py
        ├── tools/                 <- export_engine.py, gen_cert.py, rtsp_probe.py, discover_cameras.py, make_test_videos.py, ...
        ├── training/              <- train_weapon/thermal/anpr/vehicle_type.py, retrain.py, enroll_faces.py, datasets_fetch.py
        ├── static/                <- camera.html (phone page), monitor.html (web dashboard)
        ├── models/                <- weapon_detector.pt, license_plate_detector.pt, registry JSONs, face/ (gallery + onnx), plate_ocr/
        ├── data/                  <- runtime state (gitignored) — see below
        └── docs/                  <- per-feature guides; docs/README.md is the index
```

### `ibvap/` modules

| Module | Job |
|---|---|
| `detector.py` | `Detector`: batched YOLO26n, per-camera ByteTrack, carry-forward between detections, the frame-scheduling rule (§6.4), pose/weapon/face/vehicle passes |
| `rtsp_capture.py` | `RtspCapture`: pulls RTSP/HTTP/webcam/file; decode thread, TCP, reconnect + stall watchdog, `_Pacer`, `_resolve_source()` hook |
| `youtube_capture.py` | `YouTubeCapture(RtspCapture)`: yt-dlp resolve, format pick, expiry, VOD loop / live |
| `ws_capture.py` | `WebSocketCapture`: phone frames; decode, normalise to 1280x720, latency + clock sync, ack credit |
| `screen_capture.py` | `ScreenCapture`: desktop grab as a camera (`mss`) |
| `uplink_tuner.py` | `UplinkTuner`: server-chosen resolution/fps/quality rung per phone (§6.3) |
| `motion_gate.py` | cheap frame-diff pre-filter in front of the GPU |
| `posture.py` | pose head on person crops + `PoseClassifier` rules incl. AIM |
| `weapon.py` | trained `gun` model, held-gun association, temporal vote |
| `face.py` / `face_events.py` | face engine / per-person-arrival burst scheduler, vote, always-capture writer |
| `anpr.py` / `anpr_events.py` / `vehicle_type.py` | plate read / arrival-idle state machine + result writer / vehicle type |
| `geofence.py` | fences, breach, loiter, crossing counts, arming schedule, close-following |
| `behaviour.py` | running + group rules on tracks, in body-heights |
| `risk_engine.py` | zone x time x behaviour score, weapon/face/fence overrides |
| `event_store.py` | SQLite events (schema v2), ULIDs, cursor reads, ack |
| `evidence.py` | append-only SHA-256 hash chain |
| `alert_forward.py` / `sinks.py` | outbound forwarder thread / webhook, syslog-CEF, MQTT sinks + wire format |
| `snapshots.py` / `imaging.py` | evidence JPEGs per camera / shared encode helpers |
| `learn.py` / `calibration.py` | continuous-learning harvest / per-camera live threshold tuning |
| `security.py` | write-access control (loopback or `X-IBVAP-Token`) |
| `display.py` | OpenCV grid for `main.py` |

### `data/` (runtime, gitignored — a fresh clone has none of it)

`events.db` (SQLite, WAL) · `hash_chain.jsonl` (evidence chain) · `streams.json` (camera list; **overrides**
`config.yaml streams:` once it exists) · `fences.json` · `api_token` · `snapshots/<cam>/` · `anpr_results/<cam>/` ·
`face_results/<cam>/` · `plates/` · `learning/` · `calibration.json`.

### What a fresh `git clone` does NOT contain

`*.pt`, `*.engine`, `*.onnx` are gitignored except `models/weapon_detector.pt`, `models/license_plate_detector.pt`
and the 17 MB `det_10g.onnx`. So a clone lacks: `yolo26n.pt`/`.engine` (`.pt` auto-downloads; the **TensorRT
engine is GPU-specific — rebuild with `python tools/export_engine.py`, ~5–10 min**), the 166 MB ArcFace
`w600k_r50.onnx` (ships in the release package only), `vehicle_type_classifier.pt`, the thermal weights, all of
`data/`, and `cert.pem`/`key.pem` (regenerate with `python tools/gen_cert.py`).

---

## 5. Environment, running, testing, packaging

### The machine and the interpreter

Windows 11 Pro, RTX 3050 Laptop 6 GB. **The interpreter that actually runs the server today is the system
Python 3.11.9** at `C:\Users\kalpa\AppData\Local\Programs\Python\Python311\python.exe` (torch 2.13.0+cu126,
CUDA available, OpenCV 5.0.0 with prebuilt FFmpeg, `yt-dlp` installed). The vendored `backend/runtime/` that the
shipped app is *supposed* to use does not exist in this checkout, and the packaged copy under
`build/windows/x64/runner/Release/backend/runtime/` has no `pyvenv.cfg` and is missing `cv2`, `numpy` and more —
it is broken (§11, item K). The Flutter console falls back to `python` on `PATH` when no bundled runtime exists.

### Run the backend (from `backend/app`; paths are relative to it)

```powershell
cd "E:\IBVAP app\backend\app"
python server.py                 # HTTP :8090 (monitor + API) and HTTPS :8443 (phones). Interactive menu on a TTY;
                                 # a non-TTY start (CI, `> log`) falls back to --mode all, dashboard
python server.py --mode all      # every config.yaml/streams.json entry (phone slots + pulled cameras)
python run_demo.py --cams 4      # server + 4 replayed test videos + browser (--no-feed for real phones, --tunnel for a public link)
python main.py --verify-chain    # verify the evidence hash chain and exit (no server)
```

* Monitor (web dashboard): `http://localhost:8090/monitor` — plain HTTP, no certificate warning.
* Phone on the same Wi-Fi: `https://<LAN-IP>:8443/camera/0` (self-signed cert; run `python tools/gen_cert.py` first).
* Phone on another network: `python scripts/tunnel.py` (cloudflared/ngrok). **That link is public and unauthenticated.**
* Stop cleanly: `POST /api/shutdown` (loopback is allowed without a token).
* Port 8080 is deliberately avoided — Steam's webhelper squats on it and silently answers instead of the server.
* Extras: `pip install yt-dlp` (YouTube cameras), `paho-mqtt` (MQTT sink), `onvif-zeep` (ONVIF discovery resolution).

### Run the console (Flutter, from the repo root)

```powershell
cd "E:\IBVAP app"
flutter pub get
flutter run -d windows           # the console; talks to http://127.0.0.1:8090 by default
flutter analyze                  # must stay "No issues found"
flutter test                     # 40 tests
.\package.ps1                    # flutter build windows --release + mirror backend/ next to the .exe (§11, item K first)
```

The console's server address, poll interval, backend dir and Python path persist in `%APPDATA%\ibvap_app\config.json`
(`lib/config/app_config.dart`). Under `flutter test` (`FLUTTER_TEST=true`) nothing is persisted.

### Run the Python tests

`pytest` is **not installed** in the system Python. Each test module runs itself and exits 1 on failure:

```bash
cd backend/app
python tests/test_geofence.py                    # one module
for f in tests/test_*.py; do python "$f" | tail -1; done      # all 23 (each prints "N/N passed")
```

State on 2026-09-21: **320 cases across 23 modules, 0 failures** (`test_alert_forward` 13, `test_behaviour` 22,
`test_detector_schedule` 9, `test_event_store` 13, `test_face_events` 22, `test_geofence` 39,
`test_uplink_tuner` 15, `test_ws_capture` 8, `test_youtube_capture` 29, plus the older posture / risk / weapon /
thermal / ANPR / motion-gate / evidence / snapshots / learn / calibration / vehicle-type / security suites).
None needs a GPU, a model or the network. **House style** for a new module: a `CASES` list, a `case(name)`
decorator, `run()` printing `PASS`/`FAIL` per case, `test_all()` for pytest, and `if __name__ == "__main__"`.
Prefer a test that you have watched **fail** against the broken code (mutate the fix, confirm red, restore).

### Shell gotchas on this machine

PowerShell has no `head`/`tail` — use `Select-Object -First/-Last`. The Windows console is cp1252, so printing a
YouTube title or any non-ASCII text needs `PYTHONIOENCODING=utf-8`. A long heredoc into `cat > file` through the
Bash tool can fail to parse; write large files with the Write tool instead. Junctions inside a tree you delete
must be removed with `cmd /c rmdir` first, or `rm -rf` follows them into the real files.

---

## 6. Architecture

### 6.1 The pipeline

```
 sources                capture threads          muxer thread            inference worker thread          outputs
 ─────────              ───────────────          ────────────            ───────────────────────          ───────
 phone (WebSocket)  ──► WebSocketCapture ─┐
 RTSP / HTTP / cam  ──► RtspCapture ──────┤      24 Hz tick, takes the    owns the Detector: motion        /stream  MJPEG mosaic (annotated)
 video file         ──► RtspCapture ──────┼─►    newest frame of every ─► gate -> batched YOLO26n ->     /status  polled at 1–2 Hz
 YouTube link       ──► YouTubeCapture ───┤      camera, re-feeding the   ByteTrack -> pose / weapon /    events.db  ─► AlertForwarder ─► sinks
 desktop screen     ──► ScreenCapture ────┘      same one until a new     face / vehicle+ANPR passes ->   hash_chain.jsonl (evidence)
                                                 one arrives; hands a       risk + fences + behaviour ->    snapshots / anpr_results / face_results
                                                 depth-1 queue to the       events, snapshots, evidence
                                                 worker (drops if behind)
```

* **Threads.** One capture thread per pulled camera; the **muxer** (`_muxer_loop`, sleeps to a 24 Hz tick, runs no
  model); the **inference worker** (`_inference_worker`, owns the `Detector` and all per-camera tracker/gate state,
  single-threaded, no locks); the **`AlertForwarder`** daemon (own SQLite connection, never on the inference
  thread); a background OCR thread (ANPR) and an encode thread (`Harvester`). Snapshots are encoded inline on the
  worker (rare, edge-triggered).
* **Graceful degradation.** If the worker falls behind, the muxer drops the stale snapshot rather than building a
  backlog: video and detection slow down while ingest stays real-time.
* **Everything batched across cameras.** One detector call for the whole batch; then one pose pass, one weapon
  pass, one face pass and one vehicle pass over *all* crops in the batch. Per-track state is keyed
  `(cam_id, track_id)` — track 1 on two cameras is two tracks.
* **No server push to the console.** The dashboard polls `/status`. (SSE exists only for the *events* feed.)

### 6.2 Camera sources

| `url` in `streams:` / `POST /api/streams` | Class | Notes |
|---|---|---|
| `ws` (or empty, `phone`, `mobile`, `browser`) | `WebSocketCapture` | a slot a phone connects to at `/cam/N` |
| `rtsp://user:pass@ip:554/...` | `RtspCapture` | TCP forced, one decode thread, reconnect + stall watchdog, credentials redacted everywhere |
| `http://ip/video.mjpg` (IP-Webcam app etc.) | `RtspCapture` | |
| `"0"` / `"1"` | `RtspCapture` (webcam) | laptop/USB camera |
| `path/to/clip.mp4` | `RtspCapture` (file) | **paced to its own fps and looped** |
| `screen` | `ScreenCapture` | the desktop as a camera (`mss`) |
| `https://youtube.com/watch?v=…`, `youtu.be/…`, `/shorts/…`, `/live/…` | `YouTubeCapture` | needs `yt-dlp` and internet |

All of them land in the `captures` dict and are indistinguishable downstream. A capture only has to satisfy the duck
type documented at the top of `rtsp_capture.py`: `read()`, `connected`, `socket_open`, `delivered_fps`, `latency_ms`,
`info()`, `frames_received`, `stop()` (plus optional `request_reconnect()`, `get_native_frame(ts)`, `STALE_AFTER`).

**Camera list is dynamic.** `POST /api/streams` adds/edits and hot-loads a camera, `DELETE /api/streams/{id}` removes
one, `POST /api/streams/probe` tests a URL, `POST /api/streams/discover` runs ONVIF WS-Discovery. Everything persists
to `data/streams.json`, which **overrides** `config.yaml streams:` once it exists. **`upsert_stream` rebuilds the entry
from a fixed whitelist** (`id, name, url, zone_sensitivity, transport, decode_fps, enabled`) and silently drops any
other key — a new per-camera field must be added there or it vanishes on every save.

**Two timing flags on a capture.** `paced` holds a source to its own frame rate (`_Pacer`): a recording demuxes at
~470 fps if left alone (measured 45.9x real time), and an HLS live stream arrives a segment at a time so an unpaced
wall-clock decode throttle delivered 0.3 fps out of 30. Pacing matters beyond the picture: the behaviour engine
measures speed per **wall-clock** second, so an unpaced replay reads every walk as a sprint. `loop_at_end` rewinds a
recording instead of reconnecting. RTSP and webcams are paced by their own socket and are left untouched.

**YouTube.** yt-dlp resolves the link; `_resolve_source()` (a hook called fresh on every reopen) returns the media URL.
YouTube serves **no muxed formats** any more — every stream is video-only, either a direct `https` MP4 or HLS — so
yt-dlp's `best` matches nothing and `pick_format()` chooses instead: H.264 first, tallest within `youtube.max_height`
(720), a direct URL for a recording (so it can rewind), HLS for live. **DASH segment manifests are rejected** (OpenCV
cannot open them). The media URL **expires in ~6 h and is bound to this machine's IP**: it is re-resolved
`refresh_margin_s` before the deadline and dropped after any failed open (a stale link fails forever). Missing yt-dlp
disables only that camera with a readable error. It is the **only source that needs the internet** (outbound only);
yt-dlp goes stale as YouTube changes — update it first if every link fails. Age-restricted / members-only / DRM
videos will not play; no audio (there never was any). Useful free test footage with people: `Od6EeCWytZo` (Shibuya
crossing), a live one: `CXYr04BWvmc` (Bay Bridge — small distant targets, few detections), a recorded one:
`aqz-KE-bpKQ`.

### 6.3 The phone pipeline (the demo path) and the latency fix

A phone opens `https://<lan-ip>:8443/camera/<id>`; `static/camera.html` does `getUserMedia`, sends a JSON `hello`,
then JPEG frames over a WebSocket. The server decodes and **normalises every frame to 1280x720**.

The reported demo failure: on a college Wi-Fi shared from the laptop's hotspot, streams showed **10–15 s latency** and
poor quality, and faces were "detected but never registered". The fixes, all in place and tested against simulated
phones over the real WebSocket (**not yet re-tested with real phones on the real hotspot** — §11, item A):

* **Server-acked credit loop.** Each decoded frame is acked; a phone may hold at most `ingest.max_in_flight` (2)
  unacked frames. A sender that outruns the link stalls instead of growing seconds of queue (`ws.bufferedAmount`
  reads ~0 while frames sit in kernel/Wi-Fi driver queues, so the phone cannot see its own backlog).
* **Server-driven adaptation.** `UplinkTuner` picks each phone's operating point from the *measured, clock-synced*
  capture→receive delay and pushes it as `tune` messages. Ladder rungs `Rung(w, h, fps, jpeg_quality, est_mbps)`:
  `(640,360,4,0.45,0.8)` survival only · `(640,360,8,0.50,1.7)` · `(960,540,8,0.55,3.3)` **start** ·
  `(1280,720,8,0.60,5.8)` · `(1280,720,10,0.62,7.6)`. Demote at >900 ms twice, panic (two rungs) at >2500 ms, promote
  after <450 ms for 10 s and only one camera at a time; a global `uplink_budget_mbps` (12) stops phones oscillating in
  lockstep. **The ladder holds 8 fps on every rung except the last-resort one** — measured: below ~6 fps a runner moves
  further than their own box width between frames and ByteTrack loses them (one runner became 11 track ids at 4 fps,
  1 at 8 fps). It gives up resolution and JPEG quality first.
* **Stale-frame drop.** Frames older than `ingest.max_frame_age_ms` (1200) are dropped before the JPEG decode, but only
  while the phone's clock is trusted (`ws_capture.clock_ok`).
* **Latency badge.** Each frame carries an 8-byte capture-ms header; `/status` and the mosaic show `NET 120 ms`, or a
  red `NET 3.2s DELAYED` at ≥ 1.5 s.
* **Why not WebRTC/MediaMTX:** documented in `backend/app/CLAUDE.md` ("Why WebSocket-JPEG"). MediaMTX is the upgrade
  path if phone CPU or bandwidth becomes the limit.

### 6.4 How the detector spends the GPU (do not regress)

* `model.stream_fps` 24 (muxer tick) and `model.detect_fps` 8 → a fresh YOLO+ByteTrack pass every 3rd tick; between
  passes each box is advanced by its last per-tick velocity, so the annotated stream still updates at 24 fps. (Velocity
  extrapolation, not a Kalman predict.)
* **Motion gate** (`motion_gate.py`): a 160x90 grey frame difference decides whether a scene changed at all; static
  cameras never reach the GPU. Safeguards so an intrusion is never hidden: `force_every` sweeps every camera at least
  once a second, `hold_frames` keeps the gate open briefly after motion stops.
* **Fixed batch shapes** (`_bucket`): cuDNN re-tunes and a TensorRT engine *cannot* change shape, so batches are padded
  to a fixed ladder (`[max_batch]` for an `.engine`, `[1,2,4,8,16]` for `.pt`). A varying-batch run measured 304 ms/pass
  against 46 ms constant. **Easy to regress.**
* **Slow-source rule** (`Detector.process_batch`, `test_detector_schedule.py`): a fixed 8 Hz sampler aliases against a
  phone delivering ~8 fps — passes alternately land on a fresh frame and on the one after next, so the step between
  processed frames alternates 1 and 2 and a runner outruns their own box. So when a camera's measured frame period is
  ≥ 0.75 of the detection interval, **every fresh frame is processed** and repeats of the same array are skipped
  (`pipeline.frames_duplicate`, counted in `gpu_saving_pct`). Fast sources (25 fps CCTV) are still thinned to 8 fps.
* **TensorRT engine:** `weights: "yolo26n.engine"`, FP16, fixed `(8,3,640,640)`, identical detections to `.pt`, batch-1
  latency 5.6 ms vs 27 ms. Hardware-specific; fall back to `yolo26n.pt` only when the engine is missing.
* Class-restricted inference (`classes: [0,2,3,5,7]`), per-class floors (`person_conf` 0.30 for recall, `vehicle_conf`
  0.40 to kill parked-car FPs), ByteTrack `lost_track_buffer` 60 detection frames (~7 s), `minimum_matching_threshold`
  0.85. A `medium` profile (`yolo26m`) is one click away.

### 6.5 The analytics

* **Posture** (`posture.py`): `yolo26n-pose` on person crops, fixed 256 px square and padded batch (a variable shape once
  made every pass re-autotune: ~3 s vs 45 ms). Rules: LYING, CROUCH, SCAN, ARMS-UP, and **AIM** — a two-handed weapon-ready
  *posture* heuristic with no view of any weapon (a person holding a phone two-handed can trip it). AIM alone raises
  **High**, not Critical.
* **Weapon** (`weapon.py`): `models/weapon_detector.pt` (YOLO26n fine-tuned on YouTube-GDD, one class `gun`, demo-grade).
  A box is `confirmed` only if conf ≥ 0.45, area ≥ 400 px², *held* (IoU with a person box, or centre inside it), and the
  same person track carried it for 3 consecutive weapon passes. Tiers: AIM+gun on the same track = `ARMED THREAT` (99),
  gun = `GUN DETECTED` (94).
* **Face** (`face.py`, `face_events.py`): RetinaFace finds faces in padded person crops; ArcFace embeds the best one and
  matches by cosine similarity against a small enrolled gallery (`models/face/gallery.json`, **3 identities — the
  teammates**, a consented mock watchlist). A single frame never decides: up to `burst_n` (5) face-bearing readings per person
  arrival, `vote_min` (3) must agree, each past `match_threshold` (0.38, precision-biased — a false "Criminal spotted"
  is worse than a missed frame). **Always-capture:** a separate `capture_score_min` bar means a blurry, side-on, backlit
  or distant face is still *saved for the operator* (it can never name anyone, so it adds no false-match risk). Output:
  `data/face_results/<cam>/`. A confirmed match → Critical, score 96, with the identity's display name.
* **ANPR** (`anpr.py`, `anpr_events.py`): plate detector on **vehicle crops**, event-triggered (arrival → a bounded number
  of follow-up reads → idle until the track disappears), OCR on a background thread, multi-frame positional vote,
  Indian format coercion (`O/0 I/1 B/8 S/5`). Output `data/anpr_results/<cam>/` with the plate, vehicle type and both
  confidences; a vehicle whose plate never resolves is still filed with `plate: null`. Vehicle type
  (`vehicle_type.py`): 8 classes (Car / Pickup / Truck / Jeep / 2-wheeler / Tanker / Van / Auto-rickshaw); jeep and
  pickup are thin in the training data (honestly noted in the notebook).
* **Virtual fences** (`geofence.py`): polygon zones (ground point = bbox bottom-centre, edge-triggered) and multi-point
  directional tripwires, normalised 0..1 per camera, edited live via `/api/fences`. Per-fence fields: `severity`,
  `loiter_after_s`, `armed_from`/`armed_to` (local time, may cross midnight), `inbound` label (an operator-declared side,
  not geodesy), `follow_window_s` (close-following). A **maturity gate** (`min_track_passes` 3) stops a one-frame
  spurious person from breaching. Per-direction **crossing counts** with daily rollover. Debounced by `exit_passes` and
  `reentry_cooldown_passes`. Overlay burned into the mosaic *and* drawn in the console/dashboard.
* **Behaviour** (`behaviour.py`, config `behaviour:`): rules in **body-heights** (person's own box height), which cancels
  perspective without camera calibration. **Running:** net displacement over a 1.0 s window ≥ 2.0 body-heights/s for 3
  consecutive passes; a single-step jump above 6 bh/s is treated as a tracker swap and restarts history. **Group:** people
  within 1.5 body-heights are linked transitively; ≥ 3 together for ≥ 3 s (2 s flicker grace). **Close-following:** a
  second person crossing a tripwire within `follow_window_s` of the first (a pattern, not a verdict; off by default,
  per-fence). All Medium, none can reach Critical. **They are unmeasured heuristics** — thresholds come from physics and
  controlled photo motion, not real border footage; `/status.behaviour.fastest` shows what people actually measure.
  A queue at a check post *will* trip the group rule (raise `min_size`/`min_duration_s` or disable there).

### 6.6 Risk and severity

Score 0–100 = zone sensitivity 40% + time-of-day 20% + behaviour 40%; `threshold_high` 50. The score alone reaches at most
**High** (`score_can_reach_critical: false`). Overrides in `risk_engine.assess`:

| Condition | Level / score |
|---|---|
| Confirmed gun **and** AIM on the same track | Critical, ≥ 99, `ARMED THREAT` |
| Confirmed gun held by a tracked person | Critical, ≥ 94, `GUN DETECTED` |
| Watchlist face match (burst-confirmed) | Critical, ≥ 96 |
| Fence breach | Follows the **worst breached fence's own `severity`** (`server.py`, `_inference_worker`): a Critical fence (the default) → Critical, ≥ 90; a High fence → High, ≥ 70; anything lower (a counting line is not a perimeter wire) leaves the camera's level alone |
| AIM posture alone (no confirmed gun) | High (`posture_aim_level: "High"`; set `"Critical"` to restore the old behaviour) |
| Running / group / close-following / loiter | Medium by default, per-rule `severity`; never Critical alone |
| Person/vehicle present | High at most |

Camera **zone sensitivity** (`zone_sensitivity` per stream) feeds the score. Posture snapshots log at "Posture" severity
and do not force Critical unless `snapshots.escalate_posture`.

### 6.7 Events, evidence and alert egress

* **Event store** (`event_store.py`, `data/events.db`, SQLite WAL, **schema v2**). Each event has a **ULID** `event_id`
  (strictly monotonic within the process, so it doubles as the feed cursor and never sorts before an already-paged
  cursor — the same-millisecond bug was real), `ts_utc`, `site_id`/`post_name` (from `config.yaml site:`), camera identity
  (`cam_id`, `cam_name`, `lat`, `lon`), and **`severity`** (`Info|Low|Medium|High|Critical`) split from **`category`**
  (`risk, intrusion, loiter, crossing, running, group, following, weapon, posture, face_match, plate`). The legacy `level`
  column is still written with its historical meaning (`Breach`, `Posture`, `Plate`) so nothing reading it broke. Acks are
  persisted with the operator name and **appended to the hash chain**.
* **Evidence chain** (`evidence.py`, `data/hash_chain.jsonl`): every record is `{timestamp, prev_hash, event, hash}` with
  `hash = SHA-256` of the rest, so editing any record breaks every later one. `verify()` recomputes from each record's own
  stored content (so old and new record shapes both verify), treats a malformed line as tampering, and ignores an
  in-flight partial line. Snapshots, ANPR results and face results are SHA-256-bound into it (`log_to_evidence`, on by
  default). Check with
  `python main.py --verify-chain` or `GET /api/evidence/verify`. **It is tamper-evident, not a blockchain** — see §11, C.
* **Forwarding** (`alert_forward.py`, `sinks.py`): one daemon thread drains `events.db` in order, **at least once, per
  sink**; a failure holds that sink's cursor (a C2 operator needs a breach before its follow-up), each sink has its own
  cursor (a dead endpoint cannot stall a healthy one), exponential backoff with jitter (2 s doubling to 300 s). An event is
  marked synced when every *enabled* sink has taken it — or when none is configured, so a standalone install cannot grow
  the queue forever. Delivered rows are kept `alerts.retention_days` (7) / `max_queue_rows` (200000); **undelivered rows
  are never dropped** unless `max_attempts` > 0.
* **Sinks:** `webhook` (JSON POST — the primary; `WebhookSink` rewrites `localhost` to `127.0.0.1` because on Windows
  `localhost` tried IPv6 first and cost ~2 s per delivery against 15 ms), `syslog` (RFC 5424 carrying ArcSight CEF, UDP/TCP),
  `mqtt` (optional, needs `paho-mqtt`). All **disabled by default**; each has `min_severity`. Deliberately not built: SMTP
  and CAP XML (said so in the code rather than half-building them).
* **Wire format** (`sinks.event_payload`, schema `ibvap.event/2` — *this is the integration contract; add fields, never
  rename*): `{schema, event_id, ts_utc, severity, category, score, site{site_id,post_name}, camera{cam_id,name,lat,lon},
  counts{persons,vehicles}, evidence{sha256}, details{}, producer}`.
* **See it work on one laptop:** `python scripts/alert_sink.py` (receiver on `:9000`, live list at
  `http://127.0.0.1:9000/`, `--fail` to demonstrate the queue building and draining), then enable the webhook sink with
  `url: http://127.0.0.1:9000/alert`. `/status.alerts` reports per-sink queue depth, last delivery and last error.

### 6.8 HTTP / WebSocket surface (`server.py`)

Mutating `/api/*` calls (POST/PUT/DELETE) require a **loopback source or the `X-IBVAP-Token` header**
(`security.py`; token in `data/api_token`, generated on first start). **All GETs, the MJPEG streams and the phone intake are
open.** `require_token: false` and `bind_host: 0.0.0.0` are the current *testing* defaults (§11, item E).

| Area | Routes |
|---|---|
| Pages | `GET /monitor` (dashboard), `/cam/{id}` and `/camera/{id}` (phone page), `GET /` |
| Video | `GET /stream` (annotated MJPEG mosaic), `GET /stream/{id}` (one camera), `POST /api/layout` (grid/focus), `WS /ws/camera/{id}` (phone intake) |
| Status | `GET /status` (the polled snapshot: `cameras, devices, pipeline, meta, anpr, face, uplink, alerts, behaviour, geofence, learning, mosaic`), `GET /meta/{id}`, `GET /devices` |
| Cameras | `GET/POST /api/streams`, `DELETE /api/streams/{id}`, `POST /api/streams/probe`, `POST /api/streams/discover`, `POST /api/reconnect/{id}`, `GET /api/mobile-info` |
| Fences | `GET/POST /api/fences`, `DELETE /api/fences/{id}` (the console re-POSTs the **whole list** on save) |
| Events (C2) | `GET /api/events` (`limit, cam_id, level, since, severity, category, since_id=<ulid>` — `since_id` is the cursor), `GET /api/events/stream` (SSE), `POST /api/events/{id}/ack`, `GET /api/events/export?from=&to=&format=csv|json` |
| Evidence | `GET /api/evidence/verify`, `GET /api/evidence/recent`, `GET /api/snapshots`, `GET /snap/{cam}/{name}` |
| Results | `GET /api/anpr/results`, `/anpr_snap/...`, `GET /api/face/results`, `/face_snap/...`, `/api/face/gallery`, `/face_gallery_thumb/{name}` |
| Models / learning | `GET /api/models`, `POST /api/switch-model`, `GET /api/learn/pool`, `/learn/thumb/{id}.jpg`, `POST /api/learn/review`, `/api/learn/calibration`, `/api/learn/retrain` |
| Control | `POST /api/shutdown` |

`/openapi.json` currently documents paths but **no response models** (§11, item H).

### 6.9 The Flutter console (`lib/`)

Windows-only, deliberately plugin-light (`http`, `qr_flutter`; **no video or webview package** — the picture is one
hand-decoded multipart MJPEG stream, `widgets/mjpeg_view.dart`, because `Image.network` cannot do multipart). Nine tabs
(`screens/home_screen.dart`): **Monitor** (mosaic + fence draw tool + layout bar + alert/fence/learning sidebar),
**Overview**, **Cameras** (`streams_panel.dart`, `add_camera_dialog.dart`), **Models**, **Snapshots**, **ANPR**,
**Faces**, **Evidence**, **Settings** (incl. backend start/stop via `services/backend_manager.dart` and Cloudflare tunnel
via `services/tunnel_manager.dart`).

* State is `ChangeNotifier`-based, no provider package: `AppState` polls `/status` (default 1 s, clamped 300–10000 ms) with
  a generation guard against out-of-order polls and link hysteresis (`degraded` keeps the picture; only `offline` swaps in
  the offline card). Cameras are untyped `Map<String, dynamic>` — there is no camera model class.
* **`models/fence.dart` keeps an `extra` map** of every key it does not know and re-emits it on save. The console saves
  fences by re-POSTing the whole list, so a client that drops unknown fields silently *erases* server-side settings; that
  bug happened (five new fence fields were wiped on every save) and is fixed and regression-tested.
* `IbvapClient` has a 5 s default timeout, overridden to 30 s for camera probing and 15 s for ONVIF discovery (the server
  spends up to 6 s on a dead RTSP host and ~25 s asking YouTube; the shared 5 s used to report slow cameras as broken).
* **The alert log is derived from `/status` edges + snapshots** (`state/alert_log.dart`), so console alerts have no
  `event_id`, and acknowledging one is **in-memory only** — it never reaches `POST /api/events/{id}/ack` (§11, item F).
* Test style (`test/widget_test.dart`): `MockClient` injected via `IbvapClient(client: mock)`; pump widgets with
  `tester.view.physicalSize`; end any test that mounts `MonitorScreen`/`MjpegView` with `await tester.pumpWidget(const
  SizedBox())` (the live MJPEG view keeps retrying against the mock). In a `firstWhere(orElse:)` over
  `List<Map<String,dynamic>>` write `<String, dynamic>{}` explicitly, or it throws at runtime while `flutter analyze` is
  silent.

### 6.10 Models and continuous learning

`models/registry.json` lists fine-tuned models; they appear in the console's MODEL row with a `+X.X mAP` badge and
hot-swap through `/api/switch-model` (`Detector.from_weights`, path-allowlisted to `models/`). `learn.py` harvests
confident, track-stable detections into `data/learning/`; the operator keeps/drops them; `training/retrain.py` fine-tunes
and registers a model; **no weights ever change on their own**. `calibration.py` tunes per-camera confidence floors and
false-positive regions live with no training. Details: `docs/CONTINUOUS_LEARNING.md`, `docs/THERMAL_NIGHT.md`.

---

## 7. Configuration cheat-sheet (`backend/app/config.yaml`)

`config.yaml` is the pristine, hand-commented seed and the best documentation of every knob — read the comments
next to a setting before changing it. Runtime edits made in the console go to `data/streams.json` and
`data/fences.json`, **not** back into `config.yaml`, and those files win once they exist.

| Section.key | What it does / why it is set that way |
|---|---|
| `site.site_id / post_name / sector / lat / lon` | stamped on every event so two posts' logs can merge (`BOP-DEMO` today) |
| `alerts.enabled`, `alerts.sinks[]` | forwarder on/off and the sinks (webhook / syslog / mqtt); every sink `enabled: false` by default; `min_severity` per sink |
| `alerts.retention_days` (7), `max_queue_rows` (200000), `backoff_base_s` 2 / `backoff_max_s` 300, `max_attempts` 0 | the *real* "offline buffer" — size it to the longest outage you expect to replay after |
| `streams[]` | seed camera list (`id, name, url, zone_sensitivity, transport, decode_fps, enabled`); overridden by `data/streams.json` |
| `youtube.max_height` 720 / `loop_vod` true / `refresh_margin_s` 300 / `resolve_timeout_s` 20 | YouTube sources (§6.2) |
| `cctv.transport` tcp / `decode_fps` 15 / `reconnect_delay` 3 / `stall_timeout` 8 / `open_timeout` 8 | pulled-camera defaults, per-camera overridable |
| `ingest.normalise_width/height` 1280x720 | every frame is rescaled to this on arrival |
| `ingest.adaptive`, `max_in_flight` 2, `start_rung` 2, `demote_ms` 900, `panic_ms` 2500, `promote_ms` 450, `promote_dwell_s` 10, `uplink_budget_mbps` 12, `max_frame_age_ms` 1200 | the phone uplink controller (§6.3). Override the rungs with `ingest.rungs` |
| `model.weights` `yolo26n.engine`, `device: "0"`, `half: true`, `image_size` 640, `max_batch` 8 | TensorRT FP16; `max_batch` is also the engine's fixed shape — **re-export the engine if it changes** |
| `model.stream_fps` 24 / `detect_fps` 8 / `motion_gating` / `motion.force_every` / `hold_frames` | muxer tick, detector rate, gate safety sweeps |
| `model.classes` `[0,2,3,5,7]`, `person_conf` 0.30, `vehicle_conf` 0.40, `max_det` 300 | detection scope and per-class floors |
| `tracker.*` | ByteTrack (`lost_track_buffer` 60, matching threshold 0.85) |
| `risk.threshold_high` 50, `score_can_reach_critical` **false**, `posture_aim_level` **"High"**, `weights` 0.4/0.2/0.4 | §2 rule 4 — do not flip without the user |
| `pose.*`, `weapon.*` (`min_conf` 0.45, `hold_frames` 3), `anpr.*`, `vehicle_type.*`, `anpr_results.*` | per-feature thresholds; each engine self-disables with one warning if its weights are missing |
| `face.*` (`match_threshold` 0.38, `burst_n` 5, `vote_min` 3, `det_score_min` 0.55, capture bars, `max_persons_per_pass` 8), `face_results.*` | watchlist recognition + always-capture |
| `geofence.*` (`exit_passes`, `reentry_cooldown_passes`, `min_track_passes` 3, `store`) | debounce and the maturity gate; per-fence fields are edited in the console |
| `snapshots.*` (`cooldown_s` 20, `on_breach/weapon/posture/loiter/running/group/following`, `escalate_posture` false) | which events save evidence JPEGs |
| `behaviour.*` (running: `speed_bh_s` 2.0, `window_s` 1.0, `sustain_passes` 3, `glitch_bh_s` 6.0, `cooldown_s` 30; group: `radius_bh` 1.5, `min_size` 3, `min_duration_s` 3.0, `cooldown_s` 60; all `severity: Medium`) | unmeasured heuristics — tune on real footage |
| `learning.*` | harvest / calibration / retrain |
| `evidence.*` | hash-chain path |
| `security.require_token` **false**, `allow_loopback_writes` true, `token_file` | write-access posture — **testing defaults** (§11, item E) |
| `network.bind_host` **"0.0.0.0"**, `lan_phone_intake` **true** | listener exposure — **testing defaults**; `"127.0.0.1"` = this machine only |

---

## 8. What is measured, and what is not

Be exact about this when writing slides or answering a judge. **Measured** means a number was produced on this
project's hardware; **assumed/estimated** means it was reasoned or copied.

### Measured

| What | Result | Where |
|---|---|---|
| 8 phone streams (720p, 24 fps) on the RTX 3050, TensorRT FP16 | 192 fps aggregate, 0 drops, GPU ~32% mean / 45% peak, 1.0 GB VRAM, ~26 W | `backend/app/CLAUDE.md` |
| Frames the GPU never saw (rate cap + motion gate) | ~78–86% | same |
| 4-stream optimisation ladder | baseline 75.6 fps → +batching 203 → +rate cap 468 → +motion gate 467 (6.2x, 82% fewer frames) | same; `scripts/benchmark.py` |
| Crowded scene (~15 people/stream) | 2 streams ≈ 115 ms/pass (in budget); 4 streams ≈ 190 ms (over the 125 ms budget → detect rate dips to ~5 fps, tracker coasts) | same |
| TensorRT vs `.pt` | identical detections; batch-1 5.6 ms vs 27 ms | same |
| ByteTrack on a fast mover | one runner = **11 track ids at 4 fps, 1 at 8 fps**; the model itself found the runner on every frame (~0.87) — the loss is the tracker | this work |
| Fixed 8 Hz sampler vs an ~8 fps phone | frame-skip 8.3% → 2.9% at ±1.5 ticks jitter, 12.4% → 6.2% at ±2.0 after the slow-source rule; the remainder is the muxer replacing frames within a tick | `test_detector_schedule.py` |
| `localhost` webhook on Windows | 2036 ms per delivery vs 15 ms for `127.0.0.1` | `sinks.py` |
| Unpaced local/YouTube recording | 45.9x real time (30 loop restarts in 4 s) → 1.0x paced | this work |
| Live HLS at 30 fps, wall-clock throttle | 0.3 fps unpaced → 8.1 fps paced | this work |
| YouTube camera end to end | live tile in ~9 s; a street-crossing clip gave up to 10 people, vehicles, risk High (72), tracked boxes drawn, events logged; the wide Bay Bridge cam gave **0 vehicles** (tiny hazy targets) | this work |
| Behaviour rules through the real server (simulated phones + photos) | running fires once per run (~3.4 bh/s), group fires at size 3, close-following fires on the 2nd crosser (1.2 s gap); all alerts reached the receiver | this work |
| Real footage through the behaviour rules | 0 events; peak tracked speed 0.27 bh/s | this work |
| Thermal model on held-out LLVIP night IR | recall 0.53 → 0.72, mAP50-95 0.365 → 0.525 | `docs/learning_reports/` |
| Tests | 320 Python cases (23 modules), 40 Flutter tests, `flutter analyze` clean, boot log 0 tracebacks | this work |

### Not measured (do not present as fact)

* **Any real IP camera / NVR / RTSP stream.** The "19 streams" figure is GPU headroom on phones/preloaded video, decode
  excluded. Real RTSP is decoded on the CPU (no NVDEC); the ~8–12 camera figure is an estimate.
* **A false-alarm rate for any detector or behaviour rule.** The "<15% at launch" figure is a target.
* **Night / low-light performance on visible-light cameras.** Only the thermal (infrared) model has numbers, and only
  on LLVIP.
* **Running/group thresholds on real border footage.** They come from physics and controlled photo motion.
* **The hotspot latency fix with real phones on the real hotspot.** Verified with simulated phones over the real
  WebSocket only.
* Long-run stability (days), memory growth, and behaviour after a real network outage.

---

## 9. Gotchas and lessons (each of these bit us once)

**Detection / tracking**
* Do not lower the phone frame rate to save bandwidth — below ~6 fps ByteTrack loses fast movers. Give up resolution and
  JPEG quality first (§6.3).
* Do not "simplify" the slow-source rule in `Detector.process_batch` back to a fixed 8 Hz limiter; it aliases (§6.4). The
  test `heavy-jitter` fails if you do.
* Behaviour rules run only on real detection passes (`sr.inferred`) — carried-forward boxes are extrapolated and would
  make everyone look perfectly smooth.
* Boxes for photo/cut-out test subjects are tighter than a real person's height, which inflates body-heights/s.
* Small distant targets are YOLO26n's weak spot; a wide-angle public camera will show few detections. It is the model,
  not the ingest.
* The tracker withholds a new track until a second overlapping match, so a person needs 2+ detection passes before
  anything downstream sees them.

**Events / alerts**
* Same-millisecond ULIDs with independent random suffixes can sort backwards; ids are monotonic on purpose. `EventStore.log()`
  must not be handed a `ts` (only the migration passes one).
* The forwarder once did a global `break` on the first blocked sink and starved healthy ones — keep the per-sink blocked
  set.
* `localhost` in a webhook URL costs ~2 s per POST on Windows; the sink rewrites it, but still prefer `127.0.0.1`.
* A YAML mis-nesting once put `following:` outside `behaviour:` and it silently did nothing — after editing
  `config.yaml`, load it and print the block.

**Console / API**
* The console saves fences by re-POSTing the whole list — any field the Dart model drops is erased server-side. New fence
  fields must be added to `fence.dart` (or survive through `extra`) *and* tested by round-trip.
* `POST /api/streams` drops any key not in its whitelist (§6.2).
* `data/streams.json` overrides `config.yaml streams:`; editing the YAML seems to do nothing once that file exists.
* A widget that has never been rendered in a test can hide layout bugs: the Add Camera dialog's footer overflowed by 30 px
  (the Save button was clipped) until a test first pumped it.
* Flutter tests that mount the MJPEG view need the `pumpWidget(const SizedBox())` teardown.

**Sources**
* YouTube: no muxed formats, DASH is unusable, links expire and are IP-bound, the very first open of a resolved URL once
  failed and the next succeeded (hence re-resolve-on-failure), and yt-dlp goes stale.
* A paced source must never be left unpaced: the picture races and behaviour speeds inflate.
* `RtspCapture.stop()` blocks up to ~3 s joining its thread — keep it off the event loop (the API uses an executor).

**Environment**
* Steam squats on port 8080; the server uses 8090/8443.
* The TensorRT engine is per-GPU; a copied `.engine` from another card will not load.
* `backend/runtime` is missing and the packaged runtime is broken (§11, item K); the system Python is what runs today.
* Deleting a tree that contains junctions: `cmd /c rmdir` the junctions first.

---

## 10. How to verify a change (and extend the system)

**Verification ladder — do as much of it as the change deserves:**

1. **Unit tests** for the module (fast, no GPU). Add tests in the house style; confirm a new test *fails* against the
   broken code before trusting it.
2. **Whole suites:** the Python loop in §5 (expect 320+ passing), `flutter analyze`, `flutter test`.
3. **Boot the real server** (`cd backend/app; python server.py > log`), wait for `/status` (200), and grep the log for
   `Traceback` / `ERROR` (expect none). Check `/status` has the keys you changed.
4. **Exercise it with video.** In-repo: `scripts/feed_test.py --src test_videos --cams N` (replays clips as phones) and
   `scripts/loadtest_mobile.py`. Free footage with people, needs internet: add a YouTube camera through `POST /api/streams`
   (§6.2 has ids), then `GET /meta/{id}` (persons/vehicles/level), `GET /status` (`devices`, `pipeline`), and grab a frame
   from `GET /stream/{id}` to *look* at the overlay. Ad-hoc phone simulators (real photos over the real WebSocket) were
   used for the behaviour and face work but lived in a scratchpad and are not in the repo — rebuilding one into
   `scripts/` is a worthwhile task (§11).
5. **Clean up after a live test:** `DELETE` any camera you added, restore `data/streams.json`, `POST /api/shutdown`.
6. **Alerts:** run `python scripts/alert_sink.py`, enable the webhook sink at `http://127.0.0.1:9000/alert`, trigger an
   event, confirm delivery; stop the sink and confirm the queue builds and drains in order.
7. **Evidence:** `python main.py --verify-chain` must pass.

**Recipes**

* *New camera source kind:* subclass `RtspCapture` (or satisfy the duck type), override `_resolve_source()` if the URL
  needs looking up, then update `_make_pulled_capture`, `_stream_type` and the probe branch in `server.py`, add a preset in
  `add_camera_dialog.dart` and a type case in `streams_panel.dart`.
* *New event category:* add it to `CATEGORIES` in `event_store.py` (and `_LEGACY_LEVEL` if it needs a legacy value), a
  snapshot reason if it saves evidence, a console `AlertKind`, and a test.
* *New behaviour rule:* `behaviour.py` (work in body-heights, read only `sr.inferred` passes), `config.yaml behaviour:`,
  `snapshots.on_<rule>`, `/status.behaviour`, tests, and default it to **Medium**.
* *New fence field:* `geofence.Fence` + `_validate_settings`, `models/fence.dart` (known keys) + `fence_editor.dart` +
  `fences_section.dart`, a Dart round-trip test, `docs/GEOFENCE.md`.
* *New sink:* subclass `Sink` in `sinks.py` (synchronous, return `(ok, detail)` — retries belong to the forwarder), register
  it in the sink factory, add a commented example under `alerts.sinks`, and a test in `test_alert_forward.py`.
* *New config knob:* put the explanatory comment next to it in `config.yaml`, read it with a default in code, and add it to
  the table in §7.

---

## 11. What is left to do — the prioritised backlog

Items are lettered because other sections refer to them. **P0** = do before the demo/submission if at all possible;
**P1** = next; **P2** = roadmap. For each: *why it matters*, *what to do*, *where*, *done when*. Nothing here is started
unless it says so. Decisions that belong to the user are marked **[decision]**.

### P0 — before the demo / submission

**A. Field-test the latency and face fixes with real phones on the real hotspot.**
*Why:* the original failure (10–15 s latency and faces "detected but never registered" on the college Wi-Fi via the
laptop's hotspot) was reproduced and fixed only against simulated phones. *Do:* laptop on Ethernet, Windows Mobile
Hotspot, 2–3 phones on `https://<ip>:8443/camera/N`; watch the on-tile `NET` badge and `/status.uplink` (rung per
camera); walk sideways / in low light / half-hidden and check `data/face_results/<cam>/` for a saved crop each time.
*Where:* `uplink_tuner.py`, `static/camera.html`, `ws_capture.py`. *Done when:* real-device latency numbers and face-capture
counts are written down (and the ladder constants retuned if reality differs).

**B. Night-time movement detection (the weakest listed capability).**
*Why:* it is one of eight named capabilities and the only one with no automatic behaviour. *Do (suggested order):*
(1) a per-camera **day/night classifier** from frame statistics (mean luma, colour saturation — IR night footage is
near-greyscale) with hysteresis; (2) act on it: either a cheap **low-light enhancement** (CLAHE/gamma) ahead of detection
for visible-light night, or **automatic selection of the thermal/night model** for IR — note the current hot-swap
(`_switch_q` → `Detector.from_weights`) appears to replace the *whole* `Detector`, i.e. it is process-wide, so per-camera
model choice may need a second model or a design change (**verify before designing**); (3) make the thermal weights
reproducible (they are gitignored — ship them in the release or document `training/train_thermal.py` + the Colab
notebook); (4) check face/ANPR behaviour at night; (5) **measure** on real night/IR clips. Fence `armed_from/armed_to`
already covers the *rule-level* "night arming"; this item is about *perception* at night. *Done when:* a night/IR clip
produces sensible detections with no operator action, and a number exists.

**C. Blockchain — the hackathon *theme* is Blockchain & Cybersecurity, and today there is only a hash chain.** **[decision]**
*Why:* a judge scoring the theme will ask what is on a chain. *What exists:* a local, append-only, SHA-256 hash-chained
evidence ledger (`evidence.py`) — tamper-evident and verifiable, but a single writer and not anchored anywhere; it is
honestly labelled that way everywhere. *Do:* periodically **anchor the chain tip** (or a Merkle root of the records since the
last anchor) to an external public ledger, as an **outbound-only, store-and-forward** job like the alert forwarder (fits the
network rule: blockchain traffic is one of the two allowed outbound kinds; works offline and catches up), plus a verify path
that checks the local chain against the anchors. Candidate mechanisms — none tried from this repo: OpenTimestamps (free,
Bitcoin-anchored, no wallet, one small proof per anchor) or a transaction on a public EVM testnet (needs a wallet/key to
guard). The choice (which chain, key handling, whether the demo may need the internet) is the user's. Keep the wording
honest: "hash-chained ledger with external anchoring", not "blockchain-based storage". *Where:* new module beside
`evidence.py` + a sink-like forwarder; `data/hash_chain.jsonl` stays the source of truth. *Done when:* an anchor exists
externally and `--verify-chain`-style verification proves a local record was in the chain at anchor time.

**D. Measure the false-alarm rate, then decide thresholds.**
*Why:* no detector or behaviour rule has a measured rate; the "<15%" figure is a target, and a real number is the most
credible thing the submission can add. *Do:* write `scripts/false_alarm_report.py` (not written) — replay recorded footage through the real pipeline
and report alerts per hour by category into `docs/learning_reports/`. Record ordinary corridor/garden movement with the team's
phones (an evening is enough); YouTube footage is convenient and repeatable but **not representative of a border**. Then set
the running/group thresholds and confirm AIM stays capped at High (already the default). *Done when:* the first honest
alerts-per-hour-by-category table exists and the docs quote it.

**E. Make the security defaults match the standalone-and-local rule.**
*Why:* `network.bind_host: "0.0.0.0"`, `security.require_token: false` and `network.lan_phone_intake: true` are *testing*
defaults, which contradicts §2 rule 2; all GETs, the MJPEG streams and the phone intake are unauthenticated. *Do:* ship
`bind_host: "127.0.0.1"` and `require_token: true` for the standalone product, and make LAN phone intake an explicit,
switchable **demo mode** (the demo needs LAN phones; a deployment does not); add `security.require_auth_for_reads`; document
that a tunnel link is public and unauthenticated. *Where:* `config.yaml`, `security.py`, `server.py` middleware, console
Settings. *Done when:* a fresh install listens on loopback only, and turning demo mode on is a deliberate, visible act.

**G. Test against a real IP camera / NVR and measure the decode budget.**
*Why:* "existing IP CCTV" is the premise of the whole problem statement, and no real camera has been used. *Do:* use the
college CCTV/NVR sub-stream (or any IP camera) with `tools/rtsp_probe.py` then a `streams:` entry; measure CPU decode load and
the practical camera count at `cctv.decode_fps` 15; exercise reconnect and the stall watchdog by pulling the cable. *Where:*
`rtsp_capture.py`, `docs/CCTV_INTEGRATION.md`. *Done when:* a measured camera count replaces "~8–12", and the docs say what
was actually tested.

**F. Close the console ↔ server loop for alerts.**
*Why:* acknowledgement is the chain-of-custody step; today it is in-memory and never persisted. *Do:* (1) enter an operator
name once per session (stored with the install — an identifier, not a login system) and `POST /api/events/{id}/ack`; (2) feed
the console from the cursored `/api/events` so each alert carries an `event_id` (today alerts are derived from `/status`
edges + snapshots and have none); (3) a **sink-health panel** from `/status.alerts` (per-sink queue depth, last delivery,
retries, last error) — pull the network and the queue visibly builds, restore it and it drains. *Where:*
`lib/state/alert_log.dart`, `lib/screens/panels/alerts_section.dart`, `lib/api/ibvap_client.dart`, tests in
`test/widget_test.dart`. *Done when:* an ack survives an app restart, names the operator, and appears in the hash chain.

**K. Make the packaged release real.**
*Why:* the standalone-product claim depends on it, and today it is broken. *State:* `backend/runtime/` does not exist in this
checkout; the packaged `build/windows/x64/runner/Release/backend/runtime/` has no `pyvenv.cfg` and is missing `cv2`, `numpy`
and more; `yt-dlp` is installed only in the system Python. *Do:* rebuild the runtime per `backend/README.md` (clean Python
3.11 venv, torch pinned to 2.13.0+cu126, `pip install -r backend/app/requirements.txt` — which now includes `yt-dlp` — then
copy `Lib/DLLs/tcl` so it is machine-independent); re-export the TensorRT engine for the target GPU; make sure the release
carries the untracked weights (`w600k_r50.onnx`, `vehicle_type_classifier.pt`, thermal weights); run `package.ps1`; **test on a
clean machine with no Python installed**. *Done when:* the .exe launches the backend and every camera kind works there.
(`build/` also holds a stale `Release-lite` tree with old copies of docs — ignore it.)

### P1 — next

**H. Harden the C2 integration surface.** *Do:* named, revocable clients in `data/api_clients.json` with scopes
(`read`/`ack`/`admin`) — the existing single token stays as an `admin` client; FastAPI `response_model`s on the event routes
so `/openapi.json` *is* the integration contract; a `docs/C2_INTEGRATION.md` with the wire format, cursor semantics and a
one-line `curl -N …/api/events/stream` demo; then the two demo surfaces already offered but not built — a **"Command Centre"
web page** that consumes the SSE feed and can ack, and a **Telegram sink** (outbound only; needs a bot token, so secret
handling matters). *Where:* `security.py`, `server.py`, `sinks.py`.

**I. Retention for evidence folders.** *Why:* the docs state a 90-day retention target that **is not implemented** —
`data/snapshots`, `anpr_results`, `face_results`, `plates`, `learning` and `hash_chain.jsonl` grow without limit; only the alert
queue in `events.db` is pruned. *Do:* configurable age/size caps with an archive step; keep the chain's records (they stay
verifiable even after the files they hash are archived); never delete undelivered alerts.

**J. Close the suspicious-activity gaps.** *Do:* **abandoned/left-behind object** (COCO already has backpack/suitcase, but
`model.classes` restricts inference to `[0,2,3,5,7]` — widening it has a cost to measure — plus a "static object + owner left"
state machine); **person leaving a vehicle** (a person track that starts inside/at a vehicle box); and the **camera
tamper/health monitor** shown in the architecture diagram but not built (blocked lens, moved camera, frozen frame — today only
up/down/idle is shown). Build each as a heuristic that defaults to Medium and cannot reach Critical alone (§2 rule 4), with
tests in `tests/` and a `snapshots` reason.

**L. Fix the documentation drift.** Root `README.md` is stale (points at `E:\IBVAP`, lists four screens and an old layout) —
rewrite it from this file. `SETUP_AND_OPERATIONS.md` says "eight tabs" (there are nine) and has no mention of alert sinks,
webhooks or event export. `docs/GEOFENCE.md`, `INTRUSION.md` and `CCTV_INTEGRATION.md` contain **no mention** (checked by
grep, 2026-09-21) of the per-fence fields (`loiter_after_s`, `follow_window_s`, severity, arming), the running/group/following
snapshot reasons, the alert egress, or the YouTube source — all three need updating. `docs/IBVAP_Technical_Specifications_1.md` ([HARD]/[TUNABLE]/[MISSING] tags) should be brought
level with §3 and §8 here. Add `docs/C2_INTEGRATION.md`, `docs/BEHAVIOUR.md`, `docs/ALERTS.md`. (`backend/app/CLAUDE.md` was
brought current in the same pass as this file.)

**M. Licensing and privacy — decisions to make before anything leaves the team.** **[decision]**
* **Ultralytics YOLO is AGPL-3.0**; the combined work inherits it unless a commercial licence is bought. It is flagged in
  `docs/IBVAP_References.md` (~lines 98–101) and unresolved — a genuine MHA procurement question, better answered than
  discovered.
* **InsightFace `buffalo_l` weights** are non-commercial research; **LLVIP** (thermal training) is non-commercial research;
  the weapon model is trained on YouTube-GDD.
* **The teammates' face photos** (`models/face/gallery_photos/`, consented, deliberately bundled — do not move them) **are in
  the public GitHub repo's history.** Decide whether the repo should be private or the history cleaned before any wider
  release; do not delete them without asking.
* **yt-dlp / YouTube:** downloading is restricted by YouTube's terms; keep it to footage the team may use, as a test/demo
  input rather than a deployed one.
* Face matches are **leads for human verification**, never verdicts; the demo watchlist is consented and mock; real
  deployment is an API call to NCRB CrPI.

### P2 — roadmap (say "roadmap", never "live")

Multi-camera **Re-ID (OSNet)** · criminal-DB matching via **NCRB CrPI** · **VLM explainable alerts** · **feed-spoofing /
replay-attack detection** · a **GIS Digital Map** with camera markers (fences stay image-space until cameras are calibrated) ·
PostgreSQL/PostGIS and multi-post central aggregation · **Docker / Jetson Orin** port (no Dockerfile or aarch64 build exists) ·
per-camera model selection · a real Kalman predict instead of velocity extrapolation · MediaMTX/WebRTC for phones if phone CPU
or bandwidth becomes the limit · promote the ad-hoc phone simulator used in testing into `scripts/`.

---

## 12. Constraints that are not tasks

* **Inference stream:** always the secondary/sub-stream at 720p (1280x720, 15–25 fps via RTSP/ONVIF), never the primary 4K
  recording stream (~4–9x compute saved, no quality loss for this use).
* **Detection range tiers — state them explicitly, as a property of optics not a flaw:** fixed 2–4 MP bullet/dome, 0–80 m,
  full analytics; long-range PTZ 80–300 m, person/vehicle reliable but ANPR/face degrade past ~150 m; thermal 1.5–8 km, presence
  only, with close/mid-range classification via the swappable thermal model. Resolution and lens decide face range, not the AI.
* **Fences are image-space.** There is no camera calibration anywhere (no lat/lon, height, bearing, FOV, homography), so a fence
  is a normalised polygon/line on one camera's view; the `inbound` label is what the person who drew it declared.
* **Re-ID** is not live; the demo uses timestamp + visual heuristics.
* **TensorRT:** only PyTorch, TorchScript and TensorRT use the Jetson GPU — other export formats are CPU-only.
* **The AGPL position is acknowledged openly**, not hidden.

---

## 13. Documentation index, history, and where to start

### Read next (in `backend/app/docs/`; `docs/README.md` is the index)

`STARTUP.md` (bring a machine up from zero) · `CCTV_INTEGRATION.md` (RTSP/NVR/ONVIF, vendor URL tables) · `GEOFENCE.md` ·
`INTRUSION.md` · `ANPR.md` · `FACE_RECOGNITION.md` · `WEAPON_DETECTION.md` · `THERMAL_NIGHT.md` · `CONTINUOUS_LEARNING.md` ·
`SCREEN_WATCH.md` · `CUSTOM_DOMAIN.md` · `AUDIT.md` · `IBVAP_Project_Context_TechStack_1.md` ·
`IBVAP_Technical_Specifications_1.md` · `IBVAP_References.md` (22 citations). Operator guide: `SETUP_AND_OPERATIONS.md`.
Backend internals and benchmark tables: `backend/app/CLAUDE.md`. Runtime build: `backend/README.md`.

### Git history (branch `main`, in sync with `origin/main` at the last refresh)

| Commit | What |
|---|---|
| `2300003` | first commit |
| `e58d329` | backend source and configuration (excluding the runtime) |
| `d153bb3` | face-recognition watchlist + `criminal_match` stability fix |
| `d36f6b7` | hardened face capture and phone uplink; alert egress, event schema v2, border behaviours |
| `a547dbd` | documentation claims corrected to what the build supports |
| `b12cefe` | geofence, alert forwarding, behaviour detection, client models (made by the user/tooling) |
| `3e33b06` | YouTube live capture, RTSP reconnects, detector scheduling, stream-management UI (made by the user/tooling) |

### How the work got here (so the reasoning is not lost)

1. **Face capture, idle-frame gating, latency** — always snapshot a face (blurry/sideways/hidden/low-light), confirm idle
   frames are not processed, fix the 10–15 s hotspot latency (ack loop + adaptive ladder + stale drop), fix faces detected but
   never registered.
2. **Audit against the problem statement** — found detection strong and everything after it thin; built event schema v2 +
   chain of custody, alert egress + C2 feed, and the border behaviours (loiter, crossing counts, arming, maturity gate).
3. **Fence console bug** — the console re-POSTed the whole fence list and erased new fields; fixed with the `extra` map.
4. **Running, group formation, close-following** — built in body-heights; live testing then exposed three side problems (the
   4 fps ladder, 8 Hz aliasing in the scheduler, the `localhost` webhook) which were fixed.
5. **YouTube as a camera** — server-side ingest so the picture is the *annotated* mosaic tile; live testing exposed the
   no-muxed-formats / DASH / expiry facts and the unpaced-playback bug, which also affected local `.mp4` sources.
6. **Problem statement re-checked (2026-09-21)** — identical; night-time confirmed as the weakest requirement.

### If you are a new session, do this first

1. Read §2 (rules) and §3 (status). 2. `git status` and `git log --oneline -5` — trust the code over this file if they
disagree, then fix the file. 3. Run the Python suites and `flutter test` to confirm a green baseline. 4. Pick from §11. The
user's framing for the audit work: build all four priorities (event schema + chain of custody, alert egress + C2, border
behaviours, credibility), make everything **demonstrable and testable on one laptop with phone browsers as cameras**, and
**measure before deciding** on the AIM heuristic. The biggest remaining scoring risks are **B (night), C (blockchain), D
(false-alarm number), E (security defaults) and A/G (real phones, a real camera)**. 5. Do not commit or push unless asked,
keep claims honest (§2 rule 10), and update this file when the truth changes.
