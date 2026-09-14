# Code audit — Sep 2026

A pass over the whole codebase for dead weight, correctness bugs, and things
that scaled badly with the stream count. What was found and what changed.

## Why it had grown messy

The project was built feature-by-feature (batched inference → phones → latency
→ weapon posture → CCTV/RTSP → ANPR). Each pass added a subsystem and left a
little behind:

- **two capture implementations** — the original `src/capture.py`
  (`StreamCapture`) and the newer `ibvap/rtsp_capture.py` (`RtspCapture`), which is
  a strict superset (normalisation, TCP, reconnect + stall watchdog);
- **a server-push channel that nothing used** — `/ws/monitor`, `monitor_clients`,
  `_broadcast`, `_log_broadcast_error`, and a per-camera-per-tick
  `json.dumps` + `run_coroutine_threadsafe` in the hot loop. The dashboard has
  polled `/status` for a long time;
- **the secondary models bolted into a per-camera loop** — the main detector is
  batched across cameras, but pose and ANPR were called once per camera inside
  `_finish_detection`, and their per-track state was keyed by bare `track_id`;
- **stale docstrings / comments** — `server.py` still described ports 8000 and
  ngrok; `tools/export_engine.py` pointed at "config.yaml line 20";
- an **empty `docs/CLAUDE.md`** and `aiofiles` in `requirements.txt` (never
  imported);
- **only one module had tests** (`posture.py`).

## Correctness bugs fixed

### Cross-camera collision in pose + ANPR per-track state  *(the important one)*

Each camera has its own ByteTrack, so `track_id` 1 exists on every camera.
`PoseClassifier` kept `_aim_streak` / `_nose_history` / `_height_baseline` keyed
by that bare id, and `AnprEngine._plates` likewise. With two or more cameras
that had people at the same time:

- `_finish_detection` ran per camera and, at the end of each camera's pass,
  flushed every classifier key **not in that camera's** live set — so camera B's
  pass wiped camera A's aim streaks every tick;
- `AIM_HOLD_FRAMES` (2) could therefore never accumulate, so the weapon-posture
  alert effectively didn't work once more than one camera was active — exactly
  the multi-phone demo scenario.

Fix: all per-track state is now keyed by **`(cam_id, track_id)`**. The flush runs
**once per batch** over the union of live keys, and only prunes keys for cameras
that were actually in this batch — a camera that was motion-gated this pass keeps
its state instead of losing it to another camera.

### ANPR readings overwritten across cameras

`AnprEngine._plates[track_id]` and `flush_track(track_id)` had the same bare-id
problem: a plate read on camera B's vehicle 1 could evict camera A's vehicle 1,
then get filtered out by the `cam_id` check in `readings_for_cam` — the plate
flickered or vanished. Same `(cam_id, track_id)` key fix.

## Performance

### Pose and ANPR are now batched across cameras

`_infer_batch` was refactored:

- `_track(cam_id, out)` does only per-camera ByteTrack bookkeeping;
- `_pose_pass(staged)` makes **one** pose forward pass over every person crop in
  the batch (`PoseEstimator.run_batch`);
- `_anpr_pass(staged)` makes **one** plate-detector pass over every vehicle crop
  in the batch (`AnprEngine.submit_batch`).

Before: with N cameras that had people, N pose `predict()` calls per detection
tick, plus N plate-detector calls every 3rd tick — each ~5–10 ms of Ultralytics
per-call overhead, all on the latency-critical inference thread. After: 1 + 1,
regardless of N. Verified end to end (`scripts/diagnose.py` green; two-video
`process_batch` run shows `run_batch` called ≤ 1× per detection pass, keys are
`(cam, tid)` tuples, no regression in the `PIPE` numbers).

`PoseEstimator.run()` / `AnprEngine.submit()` are kept as thin single-camera
wrappers over the batched entry points.

### Dead broadcast path removed from the hot loop

`_inference_worker` no longer builds `_camera_meta` JSON and schedules a
cross-thread coroutine per camera per tick. `/ws/monitor` and the ~50 lines
behind it are gone.

### EventStore

`PRAGMA journal_mode=WAL` + `synchronous=NORMAL` (a reader never blocks the
detection thread's writes; an unclean shutdown replays instead of corrupting),
an index on `ts` and a partial index on unsynced rows, and a `close()` that
checkpoints the WAL — called from the server lifespan alongside
`AnprEngine.stop()`.

## Cleanup

| change | |
|---|---|
| `src/capture.py` deleted | `main.py` now uses `RtspCapture` — one capture stack for both entry points |
| `/ws/monitor`, `monitor_clients`, `_broadcast`, `_log_broadcast_error`, `_loop`, `broadcasts` counter | removed from `server.py` |
| `from fastapi.responses import …` inside ~8 endpoints | hoisted to the module import block |
| `server.py` module docstring, `HTTP :8080` comment | corrected (ports 8090/8443, tunnel.py, no `aiofiles`) |
| `tools/export_engine.py` next-step text | no longer references a fixed line number |
| `docs/CLAUDE.md` (empty, tracked) | removed |
| `requirements.txt` | dropped `aiofiles`; `supervision>=0.30` (the ByteTrack kwargs the code passes) |
| `EvidenceChain.append` | wrapped in a lock — matches `EventStore`'s contract if a second producer is ever added |
| `.gitignore` | `*.db-wal` / `*.db-shm` |

## Tests added (pure Python, no GPU / model / network)

- `tests/test_risk_engine.py` — scoring weights, weapon force-Critical, zone sensitivity, thresholds
- `tests/test_motion_gate.py` — skip / motion / `force_every` sweep / `hold_frames` tail
- `tests/test_evidence.py` — hash-chain append, reopen continuity, tamper + deletion detection
- `tests/test_anpr.py` — Indian-plate character coercion, per-camera reading isolation

Run any file directly, or `pytest tests/`. `tests/test_posture.py` (12 cases) unchanged and still green.

## Known limitations left as-is (documented, not bugs)

- The **TensorRT engine has a fixed batch of `max_batch`** — a single moving
  camera still costs one full `max_batch` pass. This is a deliberate trade
  (a dynamic engine reserved ~4.6 GB on a 6 GB card and stalled on shape
  changes — see `tools/export_engine.py`).
- Weapon detection is a **2D-skeleton posture heuristic**, not an object model.
- ANPR is **single-frame OCR** with no plate super-resolution or multi-frame
  voting — good for a demo, not production. The tamper-evident evidence record
  is the durable part.
- `_camera_meta` / `_latest_bgr` are written by the worker thread and read by
  `/status` without a lock — safe in CPython because each entry is replaced
  wholesale (a reader sees the old dict or the new one, never a torn one).
