# Face recognition / criminal watchlist

When a person is tracked, IBVAP tries to capture their face and checks it
against a small curated gallery of known identities — the same
detect → embed → nearest-neighbour shape as an attendance kiosk, run against a
**watchlist** instead of an allow-list. A confirmed match does not grant
anything; it raises a **Critical** "Criminal spotted" alert.

- **Model:** [insightface](https://github.com/deepinsight/insightface)
  `buffalo_l` — RetinaFace-10GF face detector (`det_10g.onnx`, ~17 MB) +
  ArcFace R50 recognizer (`w600k_r50.onnx`, ~166 MB), both committed under
  `models/face/models/ibvap_face/`. Runs on `onnxruntime-gpu` — the same CUDA
  execution provider ANPR's `fast-plate-ocr` OCR backend already uses, so no
  new runtime dependency category.
  **License:** InsightFace's pretrained model-zoo weights (`buffalo_l`, and
  every other pack) are released for **non-commercial research use only**.
  Fine for a demo; a commercial deployment would need a custom-trained or
  permissively-licensed recognizer instead — every mature InsightFace pack
  carries the same restriction, so swapping packs doesn't avoid this.
- **Data:** an **enrollment gallery**, not a training set — 2-4 clear photos
  per identity, one folder per person. See "Getting the data + enrolling"
  below. This is a one-time (or occasional) setup step, not something the
  pipeline learns from continuously.
- **Runtime:** `ibvap/face.py` `FaceEngine` runs one batched face
  detect+embed pass over padded person crops — the same shape as the pose /
  ANPR / weapon passes. It never touches the person/vehicle detector or its
  mAP, and only fires for person tracks still inside their capture burst
  (below) — a person who has already been checked costs nothing on later
  ticks.

## How a detection becomes an alert (the calibration)

A single frame's face match is **not** trusted on its own. `ibvap/face_events.py`
`PersonArrivalTracker` promotes it to a confirmed watchlist match through four
steps:

1. **Arrival** — the first tick a person track is seen opens a capture
   "burst" budget.
2. **Burst** — up to `face.burst_n` (**5**) face-bearing frames are collected
   for that track (a tick where no usable face was found in the crop doesn't
   count against the burst, only against the attempt/time budget —
   `face.max_attempts` / `face.attempt_window_s`).
3. **Match** — each face-bearing frame's 512-d ArcFace embedding is compared
   to every gallery identity by cosine similarity; a reading counts as a
   "vote" for an identity only if its similarity is ≥ `face.match_threshold`
   (**0.38**).
4. **Confirm** — the burst finalizes once `face.vote_min` (**3**) of the
   (≤5) readings agree on the *same* identity — or the attempt/time budget
   runs out first, in which case the arrival is "unknown". A single
   lucky/adversarial frame is never enough on its own; this is face's
   counterpart to the weapon detector's `hold_hits`/`hold_window` temporal
   vote.

A finalized burst — matched or "unknown" — is always filed to
`data/face_results/<cam_id>/` (the best face crop + a person-context crop + one
JSON record); only a confirmed watchlist match is also hash-chained into
evidence.

A burst that lapses its `attempt_window_s` while off the capture list — because
the window expired, the motion gate closed, or the person left frame mid-burst
— is finalized by a sweep (`PersonArrivalTracker.sweep_expired`) rather than
left pending. Without that sweep such a burst never produced a verdict at all:
`wants_capture()` and `record()`'s finish test are exact complements on the time
window, so the tick the window lapsed the track simply stopped being offered and
`record()` was never called again. A person standing still in front of a camera
reached that branch routinely, which is why a live demo logged nothing.

## Always keeping a picture of the person

Every person on a capture tick is photographed, whether or not they can be
recognised. This is deliberately **two separate bars**:

| bar | key | gates |
|---|---|---|
| `capture_score_min` (0.20) | save | is this crop worth keeping at all |
| `det_score_min` (0.55) | vote | may this reading name someone |

A face that is blurry, side-on, backlit or distant falls between them: it is
saved for the operator but marked `trusted: false` and can never vote, so
always-capture adds **no** false-match risk. Below `capture_score_min` there is
no detectable face at all, and `head_box_fallback` crops the head region
estimated from the YOLO person box instead (recorded as
`face_source: "head_box"`), so someone who never turns toward the camera is
still on file rather than invisible.

Each burst keeps only its **best** candidate, scored on detector confidence,
face pixel area, sharpness (log-normalised Laplacian variance) and frontality.
An actually-detected face always outranks a head-box guess regardless of score.
Nothing is written until the burst finalizes, so one arrival costs two small
crops (~25 KB) instead of a full-resolution frame per burst tick (~1 MB).

### Fusion tiers (`ibvap/risk_engine.py`)

`RiskAssessment` gains `criminal_match` and `criminal_name`. This is
deliberately **not** the same thing as "a person was detected" —
`risk.score_can_reach_critical` stays `false`, and a person with no face match
still tops out at High. A confirmed watchlist match is a separate, explicit
override, the same category as the weapon/gun override and a virtual-fence
breach:

| signal | level | score | banner |
|---|---|---|---|
| person present, no weapon, no watchlist match | High (at most) | ≤ threshold_high + margin | — |
| burst-confirmed watchlist match | Critical | ≥ 96 | `CRIMINAL SPOTTED: <name>` |
| watchlist match **and** an armed threat (same or different track) | Critical | 99 (armed's floor wins via `max()`) | both banners stacked |

Every confirmed match writes a `{"type": "face_match"}` row to the SHA-256
hash-chained evidence log, the same override shape as a confirmed weapon.
`python main.py --verify-chain` still passes.

## Getting the data + enrolling

The enrollment photos are **bundled inside the app**, one subfolder per
identity, and `package.ps1` carries them into every release — so the watchlist
can be rebuilt on whatever machine the product is installed on, with no
dataset folder to mount. (Unlike the internet-downloaded training datasets,
which stay outside the repo under `face.dataset_root`, these are a few hundred
KB of operator-supplied photos that define the watchlist itself.)

```
models/face/gallery_photos/
    adrika/*.jpeg
    alice/*.jpeg
    promita/*.jpeg
```

```bash
python training/enroll_faces.py --dry-run     # scan + print photo counts, write nothing
python training/enroll_faces.py               # full enroll from the bundled photos
python training/enroll_faces.py --src D:/other/photos \
    --out models/face/gallery.json --thumbs-dir models/face/thumbs
```

**Adding someone to the watchlist:** create
`models/face/gallery_photos/<name>/`, put 2–4 clear, frontal, single-person
photos of them in it, re-run `python training/enroll_faces.py`, and restart
the server.

Live matching reads only the embeddings in `models/face/gallery.json`, never
the photos themselves — the photos are what that file is rebuilt from.

Each photo must show **exactly one** clear face — a photo with zero or more
than one detected face is skipped with a warning rather than silently
guessing which face to enroll. Per-identity embeddings are the L2-renormalized
mean of each accepted photo's L2-normalized 512-d ArcFace embedding (the
standard "gallery centroid" practice for this embedding family). The first
accepted photo of each identity is copied to `models/face/thumbs/<id>.jpg` so
the console's read-only gallery listing (`GET /api/face/gallery`) never needs
`dataset_root` mounted at runtime.

Restart the server (or hot-swap the detector) to pick up a new
`models/face/gallery.json`.

## Config (`config.yaml` `face:` / `face_results:`)

| key | meaning |
|---|---|
| `enabled` | master switch. If the gallery, the two ONNX weights, or the `insightface` package are missing, the engine self-disables and the pipeline is unaffected. |
| `gallery` | `models/face/gallery.json` — embeddings + names, written by `training/enroll_faces.py` |
| `model_root` / `pack_name` | contain `models/<pack_name>/*.onnx` — the committed `buffalo_l` det+rec weights |
| `dataset_root` | where enrollment photos live — **outside the repo** |
| `det_size` | RetinaFace detector input side, px (320) — a padded person-box crop, not a full frame, so this stays small and cheap |
| `det_score_min` | discard a detected face below this confidence (0.55) |
| `match_threshold` | cosine-similarity floor for a single reading to count as a vote (0.38) |
| `burst_n` | how many face-bearing frames to collect per person arrival (5) |
| `vote_min` | of the (≤`burst_n`) readings, how many must agree on the same identity to confirm a match (3) |
| `max_attempts` | give up on this arrival after this many detection ticks even if fewer than `burst_n` faces were ever found (15) |
| `attempt_window_s` | ... or after this many seconds, whichever comes first (8.0) |
| `hit_ttl_s` | keep drawing/alerting a finalized match this long between bursts (4.0 s) |
| `detect_every` | new capture attempts run every Nth detection pass (2) — spreads GPU cost when several people arrive at once; reading back an already-finalized match happens every pass regardless, so a confirmed match's box/banner never flickers on a skipped tick |
| `crop_pad_frac` | pad the person box this much before cropping for face detection (0.25) — a tight person box often clips the top of the head/chin |
| `use_native_frame` | prefer the camera's native-resolution frame for the crop when available (true) |
| `max_persons_per_pass` | cap crops sent to the face model per camera per pass (8) |
| `face_results.dir` | `data/face_results` — one burst of pictures + one JSON record per person arrival, separate from `data/snapshots/` and `data/anpr_results/` |
| `face_results.log_to_evidence` | only a confirmed watchlist match is hash-chained; every finalized arrival (including "unknown") is still filed for audit |

## Not built this pass

In-console gallery management — adding or removing a watchlist person by
photo upload from the Flutter app — is deliberately out of scope. `GET
/api/face/gallery` is **read-only**; enrolling a new identity is
`training/enroll_faces.py` + a restart/hot-swap, the same way a fine-tuned
detector model is swapped in today. There is no existing multipart-upload
plumbing anywhere in the app to build this on top of, and it wasn't needed to
satisfy the feature this pass covers.

## Honest limits

- `buffalo_l` is sensitive to extreme pose, occlusion, and low light — a
  person turned away or far from the camera may simply never yield a
  face-bearing reading, and their burst finalizes as "unknown" rather than
  wrongly matching anyone.
- The bundled `test_videos/cam00..03` clips show people at odd or masked
  angles and are **not** a reliable way to test face recognition specifically
  — verify with a live webcam (`streams:` `url: "0"`) instead.
- A 3-identity gallery has near-zero risk of confusing the enrolled people
  with each other (embeddings self-identify at 0.76-0.96 cosine similarity in
  testing); a genuinely unfamiliar face could still occasionally cross
  `match_threshold` on enough frames to hit `vote_min` — raise
  `match_threshold` and/or `vote_min` if that's observed on real footage.
- **Working range over a phone stream is roughly 5-6 m, marginal at 10 m.**
  `static/camera.html` downscales to `SEND_W_MAX = 960` px on the longest edge
  and JPEG-encodes at q0.55 (q0.40 when the uplink congests) before sending, so
  the face the recognizer sees is far smaller than the handset's sensor
  captured. Measured on the enrolled gallery at that exact quality:

  | subject framing | face size | cosine similarity | result |
  |---|---|---|---|
  | ~3 m, doorway | ~41-50 px | 0.71-0.83 | reliable |
  | ~5-6 m, room/gate | ~27-38 px | 0.55-0.70 | reliable |
  | ~10 m, corridor | ~20 px | 0.34-0.48 | marginal — straddles the 0.38 threshold |
  | ~15 m+ | — | — | no face detected at all |

  The limiter is face *detection*, not recognition: wherever a face is found at
  all, the identity is confident. Raising `det_size` does **not** help (320 →
  800 was measured at identical accuracy for 2x the time, because the crop is
  already upscaled past the point of adding information) — the lost pixels are
  gone before the frame reaches the server. The lever that works is **send
  resolution**: 960 → 1280 lifted similarity by ~0.10-0.14 across the board and
  turned the marginal 10 m case into a solid match. JPEG quality is not the
  problem; pixels on the face are (q0.55 → q0.40 costs only ~0.03-0.07).

  Send resolution is no longer a fixed constant — `ibvap/uplink_tuner.py`
  chooses it per camera from measured latency (`ingest.adaptive`). Range
  depends directly on pixels, so it varies with link quality — the rung in use
  is drawn on the video overlay next to the latency, and is in `/status` under
  `uplink`. The ladder trades resolution and JPEG quality for bitrate but
  **holds 8 fps** on every rung except the last-resort one. Face recognition
  alone would tolerate far fewer frames, but the tracker would not: below ~6 fps
  it loses a person who is running (measured: one runner became 11 track ids at
  4 fps, 1 at 8 fps), and running detection and tripwire crossings depend on
  tracks.
- **No liveness / anti-spoof check.** A printed photo or a phone/tablet
  screen held up to the camera can also match. This is a known gap, not
  addressed this pass — a real security deployment needs a liveness check
  before trusting a face match for anything more consequential than an alert
  an operator reviews.
