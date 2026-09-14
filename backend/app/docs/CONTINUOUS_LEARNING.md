# Continuous learning

IBVAP improves on its own deployment over time. Two things run continuously; a
third is a deliberate, operator-triggered step.

**Nothing changes model weights silently.** "Continuous" means the data
collection and the per-camera calibration never stop — the actual fine-tune is a
command you run and a model you choose to switch to (and can roll back).

## 1. Harvest (always on)

`ibvap/learn.py` `Harvester` — hooked into `server._inference_worker` (and into
`screen_watch.py --learn`). On every real detection pass it looks at each
detection and keeps a training example only when the box is **both**:

- **confident** — `conf >= learning.harvest.min_conf` (0.55), and
- **track-stable** — the same `(cam_id, track_id)` has been seen for
  `min_track_frames` (5) consecutive inferred passes.

A frame is skipped entirely if any box sits in the "not sure" band
`[ambiguous_conf, min_conf)` — a half-seen car you don't label teaches the model
"no car here". Harvesting is rate-limited (`per_cam_rate_limit_s`, 3 s/camera),
content-deduped, and capped at `max_pool` (5000) pending items.

Each kept frame is written as a YOLO example:

```
data/learning/frames/<id>.jpg      the raw normalised 1280x720 frame
data/learning/labels/<id>.txt      "<cls> <cx> <cy> <w> <h>" normalised, one line per box
data/learning/thumbs/<id>.jpg      ~320px preview for the dashboard
data/learning/manifest.jsonl       one append-only row per candidate
data/learning/verdicts.jsonl       one append-only row per operator decision (last wins)
```

The write (JPEG encode + files) happens on a daemon thread; the hook on the
inference worker is a few dict lookups and one `frame.copy()` gated behind the
rate limit — no measurable fps cost.

Classes are **not** remapped: the pseudo-label file only ever contains COCO ids
`0,2,3,5,7` (person + 4 vehicle types), and `training/retrain.py` writes `data.yaml` with
`nc: 80`, so the fine-tuned head has the same shape as `yolo26n.pt` and the
hot-swap is drop-in.

## 2. Per-camera calibration (always on, no training)

`ibvap/calibration.py` `Calibrator` — owned by the inference worker, threaded
through model hot-swaps like `anpr=`. It watches what each camera's detector
actually does and adjusts two things per camera, live:

- **confidence floor** — raised where confident-but-transient detections keep
  appearing and dying without ever forming a track; lowered (only within
  `headroom`, 0.05) where real objects sit just under the bar. Bounded to
  `[-headroom, +max_tighten]` (max +0.30).
- **suppression mask** — a coarse `32x18` grid; a cell where detections
  repeatedly appear-and-die (`>= promote_cell_after`, 25) becomes "known noise"
  and *weak* boxes there are dropped (a confident intruder is never suppressed —
  the check is `AND conf < bar + 0.10`).

Nothing applies until `min_obs_before_apply` (300) observations for that camera,
so it can't over-fit in the first few seconds. Old counts halve every
`decay_per_day`. State persists to `data/calibration.json`; from the dashboard
you can `reset` / `freeze` / `set` a camera's deltas
(`POST /api/learn/calibration`).

The applied floors / suppressed cells feed the pre-tracker filter in
`Detector._track` — the batched forward pass is untouched.

## 3. Fine-tune + hot-swap (deliberate)

`training/retrain.py` — CLI, also spawned by the dashboard **Retrain** button
(`POST /api/learn/retrain` → `subprocess.Popen`, stdout tailed into
`/status.learning.last_train`).

```
python training/retrain.py            # detector fine-tune from the kept pool
python training/retrain.py --dry-run  # assemble the dataset + print counts, no training
python training/retrain.py --posture  # threshold-sweep report vs. operator verdicts
python training/retrain.py --anpr     # plate-detector data report
```

Detector mode:

1. `assemble_dataset` — `keep` items → `datasets/ibvap_live/images,labels/{train,val}`
   (split **chronologically**, no near-dup leakage); `background` items → an
   empty label file (hard negative); operator-corrected boxes override the
   pseudo-label. Refuses below `retrain.min_images` (20).
2. `write_data_yaml` — `nc: 80`, full COCO names, absolute `path:`.
3. `YOLO("yolo26n.pt").train(..., freeze=10)` — backbone frozen + a slice of
   COCO-val mixed in (`replay_images`) to limit catastrophic forgetting.
   `yolo26n.pt` only — a `.engine`/`.onnx` cannot be trained.
4. `evaluate` — `model.val(classes=[0,2,3,5,7])` for the base and the fine-tuned
   model on the held-out split → **base vs fine-tuned mAP50 / mAP50-95**.
5. `register` — copies `best.pt` → `models/yolo26n_ft_<ts>.pt`, appends
   `models/registry.json` (the only tracked artifact — small JSON).

The fine-tuned model then appears in the dashboard **MODEL** row with a
`+X.X mAP` badge (`GET /api/models` merges the registry). Click it to hot-swap
(the existing `_switch_q` path, which now carries a weights dict as well as a
profile name); click **Nano** to roll back. Rollback = switch to `nano`.

**Honesty:** `freeze=10` + few epochs + COCO replay is *mitigation*, not a
guarantee against forgetting. The base-vs-fine-tuned mAP on the held-out split
is the check — a drop on the base classes means don't ship it. On a demo the
pseudo-labels are the model's own output, so mAP starts ~1.0 and barely moves;
the improvement shows when the operator *corrects* boxes and marks backgrounds.

## Demo script

1. `python server.py` + `python scripts/feed_test.py --cams 2 --src test_videos`.
   Open `/monitor` — the **LEARNING** panel's `harvested` counter ticks up; the
   thumbnail strip fills.
2. Point a camera at a scene with a recurring false box in one corner. After
   ~300 observations the panel shows a calibration delta and a `■` suppression
   count for that camera and the false box stops being drawn.
   `POST /api/learn/calibration {"cam_id":0,"action":"reset"}` clears it.
3. In the panel: **Keep** ~25 good crops, **Bg** a couple of empty frames.
4. Hit **Retrain**. `last_train` goes `running → done` with a Δ mAP.
5. A new **FT …** button appears in the MODEL row. Click it → the detector
   hot-swaps. Click **Nano** → rolls back. `python main.py --verify-chain`
   still `Chain OK` (evidence is untouched).
6. `python scripts/screen_watch.py --pick --learn` harvests from a screen region too.

## Config (`config.yaml` `learning:`)

| key | meaning |
|---|---|
| `enabled` | master switch for harvest + panel |
| `harvest.min_conf` / `min_track_frames` / `ambiguous_conf` | the pseudo-label gate |
| `harvest.per_cam_rate_limit_s` / `max_pool` | keep the pool from exploding |
| `calibration.enabled` / `headroom` / `max_tighten` | live floor adjustment bounds |
| `calibration.promote_cell_after` / `min_obs_before_apply` / `decay_per_day` | the suppression mask + when it applies + how it fades |
| `retrain.epochs` / `freeze` / `val_fraction` / `replay_images` / `min_images` | the fine-tune |

`tests/test_learn.py` + `tests/test_calibration.py` cover the pseudo-label gate,
rate limit, label format, EMA bounds and suppression-cell promotion (no GPU).
