# Virtual fences (digital fencing)

The operator draws **polygon zones** and **directional fence lines** (2+ points,
polyline — each segment is a tripwire) on each camera's own view in the
dashboard FENCES panel. When a tracked person or vehicle breaches a fence, the
camera is forced to **CRITICAL** and the breach is written to the SHA-256
hash-chained evidence log — the same override path the weapon-ready posture uses.

## No GIS projection — and why

The problem statement asks for a "GIS Digital Map", but IBVAP has **no camera
calibration** anywhere: no lat/lon, mounting height, bearing, field of view, or
ground homography (`config.yaml`, `ibvap/rtsp_capture.py`, `tools/discover_cameras.py`).
Image-space fences therefore cannot be projected onto real-world coordinates.

So a fence is **normalised 0..1 image geometry attached to a `cam_id`** — it
lives in the camera picture, not on a map. Normalised coordinates mean a fence
survives a change of `ingest.normalise_*` or camera resolution. A marker map
(one pin per camera, coloured by live risk) is spec'd as a later add-on in the
implementation plan but is deliberately **not** built — it would be decoration,
not the digital-twin projection the slogan implies.

## Pipeline

```
detector (8 fps) ─► StreamResult ─► server.py _inference_worker, per camera:
     ra = risk_engine.assess(sr)
     breaches = geofence.evaluate(sr)          # ibvap/geofence.py, pure geometry
     if geofence.active_zones(cam):  ra.level, ra.score = "Critical", max(ra.score, 90)
     per NEW breach → evidence.append({type:"breach",...}) + store.log(cam,"Breach",...)
     _camera_meta[cam] += {breach, breach_zones}
     _draw_fences(frame_out, ...)              # green / filled-red overlay on the mosaic
                                               │
monitor.html poll(/status) ─► SVG overlay recolour + BREACH row in the alert log
```

`ibvap/geofence.py` has no GPU / model / OpenCV dependency and is covered by
`tests/test_geofence.py` (run `python tests/test_geofence.py`).

### Breach semantics

| kind | fires when | notes |
|---|---|---|
| **polygon** (zone) | the target's **ground point** — bbox bottom-centre, i.e. where it stands — enters the area | edge-triggered: one breach on entry. A target hugging the boundary is not re-fired every pass — `geofence.exit_passes` consecutive "outside" reads end the breach. |
| **line** (tripwire / fence line) | the segment *previous ground point → current ground point* crosses **any** segment of the fence, in the flagged `direction` | 2 or more points — each consecutive pair is a segment, so a bent border fence is one fence. `direction`: `a2b` \| `b2a` \| `both`, relative to each segment's own A→B orientation (`a2b` = from the positive side of A→B to the negative side; the dashboard and mosaic draw per-segment arrows so the operator picks it visually). One breach per fence per pass, then debounced by `geofence.reentry_cooldown_passes`. |

`targets` (`any` \| `person` \| `vehicle`) gates which classes a fence watches.
Per-track state is keyed `(cam_id, track_id)`, so track id 1 on two cameras
stays two independent tracks; `geofence.prune()` drops state for track ids a
camera no longer reports (mirrors `PoseClassifier.flush_missing`).

## Config (`config.yaml` `geofence:`)

| key | meaning |
|---|---|
| `enabled` | master switch |
| `store` | JSON file the dashboard reads/rewrites (default `data/fences.json`, gitignored) |
| `eval_on` | `detect` = only real detection passes (8 fps); `track` = every frame incl. carried |
| `reentry_cooldown_passes` | a tripwire won't re-fire for the same track within this many passes |
| `exit_passes` | consecutive "outside" passes before a zone breach is considered over |

## HTTP API

| method | path | body / result |
|---|---|---|
| `GET` | `/api/fences[?cam_id=N]` | `{"fences": [...]}` |
| `POST` | `/api/fences` | `{"fences": [ ... ]}` — **full-set replace** (the dashboard holds the whole list). Validated; `400 {"errors": [...]}` on bad geometry. Missing `id` / `created_at` are filled in. Persisted atomically, then handed to the worker over `_fences_q`. |
| `DELETE` | `/api/fences/{id}` | drops that fence, rewrites, reloads |

### Fence object

```json
{"id": "f_1a2b3c", "cam_id": 0, "kind": "polygon",
 "points": [[0.12,0.80],[0.44,0.62],[0.71,0.90]],
 "direction": "both", "targets": ["person","vehicle"],
 "label": "Gate apron", "enabled": true, "created_at": 1725800000.0}
```

`points` are normalised 0..1; polygon ≥ 3, line ≥ 2 (a multi-point line is a
polyline — each consecutive pair is a tripwire segment). `direction` is line-only.

## Dashboard

`static/monitor.html`, FENCES panel:

1. Pick a camera (or click its mosaic tile — same math).
2. Choose **Zone** / **Line**, target class, and (line only) direction.
3. Click **Draw**, then click points on that camera's tile. **Undo pt** / **Cancel** as needed.
4. Type an optional label, click **Save** → `POST /api/fences` with the whole set.
5. Fences show as an SVG overlay on the mosaic — green normally, red while breaching.
   The list below has an enable checkbox and a delete "×" per fence.

The click → tile → `cam_id` + pixel math undoes the mosaic's `object-fit: contain`
letterboxing and uses `/status.mosaic` (`ids`, `cols`, `tile`) so it always
matches the exact grid `server._build_mosaic()` composed.

## Output

- **Overlay** — burned into the mosaic (`_draw_fences`) *and* an SVG layer in the dashboard.
- **`/status.geofence`** — `{enabled, fences: {cam: [...]}, active: {cam: {fence_id: bool}}}`.
- **`_camera_meta[cam].breach` / `.breach_zones`** — per-camera, in `/status.meta`.
- **Evidence chain** — one `{"type": "breach", ...}` record per new breach; `main.py --verify-chain` stays valid.
  The record now also carries `evidence_file` + `sha256` pointing at the snapshot (below).
- **`data/events.db`** — a `level="Breach"` row per new breach.
- **Alert log** — a red `BREACH` row per camera-level breach edge, with the snapshot thumbnail.
- **Snapshot** — a JPEG of the exact frame is saved under `data/snapshots/<cam_id>/`
  (annotated + raw), with the image SHA-256 bound into the breach record. Same
  mechanism captures a confirmed weapon or a crouch/lying posture. Full detail:
  **`docs/INTRUSION.md`**.

## Seeding a demo

`data/` is gitignored. Copy the tracked seed and start the server:

```bash
cp fences.example.json data/fences.json
python server.py
```

Or just draw fences from the dashboard — `POST /api/fences` rewrites that file.
