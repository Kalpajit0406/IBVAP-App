# ANPR — number-plate recognition

Two stages, both now tunable for Indian plates:

- **Plate detector** — `models/license_plate_detector.pt`. Originally the
  YOLOv8n from [anindya-mukhopadhyay/ANPR](https://github.com/anindya-mukhopadhyay/ANPR)
  (MIT); **`training/train_anpr.py --detector`** fine-tunes `yolo26n` on Indian
  licence-plate datasets and replaces it in place.
- **OCR** — backend `anpr.ocr_backend`: **`fast_plate`**
  ([fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) — a
  plate-specific ONNX recogniser, `cct-s-v2-global-model`, no torch at runtime),
  falling back to **`easyocr`** (generic English + heuristics) when it isn't
  installed. `training/train_anpr.py --ocr` (and the Kaggle notebook) fine-tune it
  on Indian crops → `models/plate_ocr/`.
- **Multi-frame voting** — a tracked plate is read many times; each character
  position is majority-voted across the track, so one blurry frame can't decide
  the plate. Backend-agnostic; on by default (`anpr.vote_min_reads`, 3).

## How it runs

```
main detector ──► vehicle boxes (car/motorcycle/bus/truck, with track ids)
                        │
                        ▼   every 3rd detection tick (~2.7 fps)
     YOLO "license_plate" on each VEHICLE CROP  (models/license_plate_detector.pt)
                        │  plate box → mapped back to full-frame coords
                        ▼   bounded queue (drop-oldest)
     OCR worker thread   fast-plate-ocr .run(crop)   (or EasyOCR upscale·CLAHE·readtext)
                        │  clean → region-format nudge (IN / XX)
                        ▼
     per-(cam,track) reading buffer  ──► majority-vote each char position (>= vote_min_reads)
                        │  score = vote agreement
                        ▼
     voted plate  ──► drawn under the vehicle box
                  ──► data/plates/plates.csv + crop image
                  ──► SHA-256 hash-chained evidence log
                  ──► dashboard alert-log row + "plates" counter
```

Everything after the vehicle boxes is **asynchronous** — the OCR worker never
touches the real-time detection budget. A vehicle that already has a confident
reading (≥ 0.55) is skipped until the cooldown elapses.

## Setup

```bash
pip install "fast-plate-ocr[onnx-gpu]"    # recommended OCR backend (light, ONNX)
pip install easyocr                        # fallback backend
```

`config.yaml` → `anpr.enabled: true` (default). If **neither** OCR backend is
installed, or `models/license_plate_detector.pt` is missing, `AnprEngine` logs
one warning and disables itself — nothing else breaks. With only `easyocr`
present it auto-falls-back and still runs (with voting).

## Training

```bash
python training/datasets_fetch.py --anpr                 # Indian plate sets → E:/Projects/ibvap-datasets/anpr
python training/datasets_fetch.py --anpr --roboflow-key KEY   # + a Roboflow plate set
python training/train_anpr.py --detector --dry-run       # assemble plate_yolo/, counts
python training/train_anpr.py --detector                 # yolo26n → models/license_plate_detector.pt
python training/train_anpr.py --ocr                      # assemble + (if fast-plate-ocr[train]) fine-tune
```

`--ocr` also folds in **operator-confirmed crops the pipeline already collects**
(`data/learning/plates/*.jpg` + `plates.jsonl`, via `Harvester.submit_plate`).
On Kaggle: **`notebooks/train_anpr_kaggle.ipynb`** (Add Data → the 3 Kaggle
slugs → Run all) produces `license_plate_detector.pt` + `plate_ocr/`. Same
outputs on **Google Colab**: **`notebooks/train_anpr_colab.ipynb`** — pulls
the same 3 Kaggle datasets via the Kaggle API (upload a `kaggle.json` token in
the notebook) instead of Kaggle's "Add Data" panel, then zips both outputs for
download.

## Config (`config.yaml` `anpr:`)

| key | meaning |
|---|---|
| `enabled` | master switch |
| `weights` | plate-detector weights (shipped in `models/`) |
| `region` | `IN` = Indian format `SS RR L(L)(L) NNNN` (e.g. `WB06AB1234`); `XX` = any 4–10 alphanumerics with a letter and a digit. A format match boosts the stored confidence by 0.15. |
| `yolo_conf` | plate-detector confidence (0.30) |
| `ocr_min_conf` | min mean OCR confidence to **log** a plate (0.20) |
| `min_plate_area` | px² floor on plate boxes (300) |
| `plate_detect_every` | run the plate stage every Nth detection tick (3) |
| `plate_imgsz` | plate-detector input size (320 — plates are small) |
| `ocr_backend` | `fast_plate` (fast-plate-ocr) or `easyocr`. `fast_plate` auto-falls-back to `easyocr` if the package is absent. |
| `ocr_model` | `cct-s-v2-global-model` (bundled) or a `models/plate_ocr` dir from `train_anpr.py --ocr`. A missing local path falls back to the bundled model. |
| `vote_min_reads` / `vote_max_reads` | start majority-voting a plate at this many readings (3); ring-buffer cap per track (12) |
| `dataset_root` | where `datasets_fetch.py --anpr` puts data — outside the repo |
| `cooldown_s` | don't re-log the same string within this window (8 s) |
| `workers` | EasyOCR worker threads (1) |

## Output

- **`data/plates/plates.csv`** — `timestamp, cam_id, track_id, plate, confidence, valid, crop`
- **`data/plates/<ts>_cam<n>_<PLATE>.jpg`** — the plate crop (best effort)
- **Evidence chain** — a `{"type": "plate", ...}` record per confirmed plate
- **`/status` `anpr`** — `plates_detected / ocr_runs / plates_logged / queue_drops` + `active` readings
- **Dashboard** — yellow `PLATE` rows in the alert log, a `plates` figure in the header

## Tuning notes

- **Missing plates?** Lower `yolo_conf` to 0.20, raise `plate_detect_every` to 2,
  make sure the camera's sub-stream is sharp enough — plates need ~80 px width
  to OCR. A main stream gives more pixels but costs CPU.
- **Wrong characters?** Indian plates: `0/O`, `1/I`, `8/B`, `5/S` confusions are
  common. The `region: IN` check rejects strings that can't be a valid plate,
  but doesn't correct them. A plate-specific OCR model (e.g. fast-plate-ocr) is
  the upgrade path.
- **CPU hot with many vehicles?** Raise `plate_detect_every`, keep `workers: 1`,
  or gate ANPR to specific cameras (a `cameras: [0, 2]` filter is a small
  addition to `AnprEngine.submit`).

## Limitations

2-D, single-frame OCR with no plate-super-resolution and no multi-frame voting.
Good for a demo ("that car's plate is WB06AB1234, logged at 14:32, here's the
crop"); not production ANPR. The evidence record is the valuable part — it's
tamper-evident and timestamped.
