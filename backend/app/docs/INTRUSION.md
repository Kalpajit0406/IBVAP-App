# Intrusion snapshots & operator layout

Two related additions on top of the virtual-fence engine (`docs/GEOFENCE.md`):

1. **Snapshots** — when an intrusion or a "malicious movement" is detected, save
   a picture of that exact moment, per camera, tamper-evidently.
2. **Operator layout** — focus on 1–4 streams with the rest on the side, or view
   every stream as a grid.

---

## 1. Snapshots (`ibvap/snapshots.py`)

### Triggers

| Reason | Fires when | Severity | Snapshot |
|---|---|---|---|
| `breach` | `GeoFenceEngine.evaluate()` returns a new `Breach` (someone's ground point enters a zone, or crosses a tripwire in the flagged direction) | forces the camera to **Critical** (unchanged) | yes |
| `weapon` | `WeaponDetector.drain_events()` yields a confirmed firearm | **Critical** (unchanged) | yes |
| `posture` | a tracked person is `crouching` or `lying` (`snapshots.malicious_postures`) | logged at **`Posture`**; does **not** force Critical unless `snapshots.escalate_posture` | yes |

`chest_aim` / `arms_up` are *not* posture triggers here — they already route
through the weapon / risk path.

### Files

```
data/snapshots/
├── 0/                                   # camera 0 — created on its first frame
│   ├── 20260911_022340_015_breach_north-gate.jpg       # annotated (boxes, skeleton, fence)
│   ├── 20260911_022340_015_breach_north-gate_raw.jpg   # untouched frame
│   └── index.jsonl                                     # one line per event
├── 1/                                   # camera 1 — created when camera 1 first streams
│   └── …
```

Folder name is `str(cam_id)` — no lookup table. One camera connected → folder
`0`; add a second → folder `1` appears; and so on. `data/` is gitignored.

Each `index.jsonl` line:

```json
{"ts": 1789073870.11, "reason": "breach", "cam_id": 0, "track_id": 5,
 "detail": "north-gate", "file": "0/20260911_022340_015_breach_north-gate.jpg",
 "raw_file": "0/…_raw.jpg", "sha256": "354c50…", "raw_sha256": "a0e007…",
 "deduped": false, "ev_hash": "5045b2a4e98f…"}
```

### Tamper-evidence

The annotated image's `sha256` is written **into** the hash-chained evidence
record for the event (`evidence_file` + `sha256` fields on the `breach` /
`weapon` / `posture` record). `ev_hash` in `index.jsonl` is that record's chain
hash. So:

* recompute the file's SHA-256 → must equal `sha256` in the record and the index;
* `python main.py --verify-chain` → must stay `Chain OK`.

Editing a snapshot, or swapping one, breaks the match; editing the chain breaks
`--verify-chain`.

### Debounce & dedup

* `SnapshotWriter.should_capture(cam_id, track_id, reason)` returns `True` at
  most once per `snapshots.cooldown_s` (default 20 s) for a given
  `(camera, track, reason)`. A track that flickers out of ByteTrack for a pass
  and returns keeps its guard — `prune()` only forgets a track's guard once it
  is *both* gone *and* past the cooldown, so a re-entering track does not
  re-snapshot every second.
* Several intruders crossing in the **same** detection pass produce one JPEG
  pair on disk (identical frame) — each still gets its own `index.jsonl` line and
  its own evidence record, with `deduped: true` on all but the first.

### Cost

Encoded **inline on the inference worker** — ~5–15 ms for one 720p frame, only on
rare edge-triggered events, well inside the ~125 ms detection budget. (The
continuous-learning `Harvester`, which fires constantly, offloads to a daemon
thread; snapshots do not, because the sha must be known in time to bind it into
the chain.)

### "Exact moment"

There is no frame ring buffer, so the snapshot is the frame of the detection
pass that *detected* the crossing — within ~1 detector tick (~125 ms) of the
line being crossed. A pre-roll video clip is out of scope.

### Config (`config.yaml` `snapshots:`)

| key | default | meaning |
|---|---|---|
| `enabled` | `true` | master switch; `false` → the writer is a silent no-op |
| `dir` | `data/snapshots` | root; per-camera folders are created under it |
| `annotated` | `true` | save the frame with boxes / skeleton / fence overlay burned in |
| `raw` | `true` | also save the untouched camera frame |
| `jpeg_quality` | `90` | |
| `cooldown_s` | `20.0` | per `(camera, track, reason)` — one snapshot per ongoing event |
| `on_breach` / `on_weapon` / `on_posture` | `true` | per-reason switches |
| `escalate_posture` | `false` | also force the camera to Critical on a crouch / lying |

### Routes (read-only)

| route | returns |
|---|---|
| `GET /snap/<cam_id>/<name>` | the JPEG (`<name>` validated, must stay inside the camera folder) |
| `GET /api/snapshots?cam_id=&limit=` | newest-first tail of the `index.jsonl` line(s) as JSON |
| `GET /status.snapshots` | `{enabled, dir, written, by_reason, escalate_posture}` |

The dashboard ALERT LOG polls `/api/snapshots` every 2 s and shows each new
snapshot as a thumbnail linking to the full image.

### Tests

`python tests/test_snapshots.py` — 11 cases, no GPU / no network: per-camera
folder, annotated + raw write, 64-hex shas, `index.jsonl` shape, `recent()`
order, `should_capture` cooldown, `prune()` flicker-vs-cooled, `on_*` gating,
disabled no-op, same-frame dedup, `malicious_postures` selection.

---

## 2. Operator layout (`server._mosaic_tiles`, `static/monitor.html`)

The mosaic stays a **single** server-composited MJPEG (`GET /stream`) — one HTTP
connection regardless of camera count. Two layouts:

* **Grid** (default) — uniform tiles, row-major by sorted `cam_id`.
* **Focus** — 1–4 chosen "main" cameras fill the left ~78 % large; every other
  camera is a small tile down a right-side filmstrip.

The layout is **global** (the mosaic is one shared image): whoever changes it
changes it for everyone watching. `POST /api/layout {"mode":"grid"|"focus",
"mains":[0,1]}` stores it in `_stats["layout"]`; the next `/stream` frame and the
next `/status` poll pick it up — no reconnect.

`/status.mosaic` describes the current composition:

```json
{"mode": "focus", "mains": [0], "canvas": [720, 270],
 "tiles": [{"cam_id": 0, "x": 0, "y": 0, "w": 480, "h": 270},
           {"cam_id": 1, "x": 480, "y": 0, "w": 240, "h": 270}],
 "ids": [0, 1], "cols": 2, "tile": [480, 270]}
```

`tiles` is the rect map the dashboard's fence-draw tool clicks against — grid is
just the special case where every rect is equal, so drawing a fence works in
either layout, including on a camera that is only in the filmstrip. The Grid /
Focus choice and the main-camera set persist in `localStorage`.

### Fence panel — set / edit / remove per stream

* **Set** — pick a camera, Zone or Line, **Draw**, click points, **Save**
  (unchanged).
* **Edit** — click a fence's name in the list: its points load back into the
  draft; reshape and **Save** replaces it by `id`.
* **Enable / disable** — the per-fence checkbox flips `enabled` and re-POSTs.
* **Remove** — the `×` deletes one fence; **Clear cam** removes every fence on
  the selected camera.
