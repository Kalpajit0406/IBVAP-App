# models/

## `license_plate_detector.pt`

A YOLO detector for a single class, `license_plate`, run on each vehicle crop
(`ibvap/anpr.py`).

- **Origin:** YOLOv8n from https://github.com/anindya-mukhopadhyay/ANPR
  (MIT © 2023 BAPPY AHMED).
- **`training/train_anpr.py --detector`** fine-tunes `yolo26n` on Indian
  licence-plate datasets and **overwrites this file** (a timestamped copy +
  `models/anpr_registry.json` keep the history). ~6 MB, committed
  (`.gitignore` `!models/license_plate_detector.pt`).

## `plate_ocr/`  *(created by training)*

The plate-text OCR for the `fast_plate` backend — a `fast-plate-ocr` ONNX model
(`model.onnx` + `config.yaml`), fine-tuned on Indian crops by
`training/train_anpr.py --ocr` / `notebooks/train_anpr_kaggle.ipynb`. Committed
when present (`.gitignore` `!models/plate_ocr/`). If absent, `ibvap/anpr.py`
uses the bundled `cct-s-v2-global-model`, then falls back to EasyOCR.

To disable ANPR entirely, set `anpr.enabled: false` in `config.yaml`.
Full guide: **`docs/ANPR.md`**.

## `weapon_detector.pt`

`yolo26n.pt` fine-tuned to a single class, `gun`. Used by `src/weapon.py` — it
runs one batched pass over each camera frame; a gun box is only alerted when it
is held by a tracked person and persists (`weapon.hold_frames`).

- **Source:** trained locally by `python train_weapon.py` on
  [YouTube-GDD](https://github.com/UCAS-GYX/YouTube-GDD) (5000 images, gun class).
- ~6 MB. Committed to the repo (`.gitignore` `!models/weapon_detector.pt`) so a
  fresh clone can demo without re-training. Re-train any time to improve it;
  `train_weapon.py` overwrites this file and appends `weapon_registry.json`.
- Not present until trained — until then `src/weapon.py` self-disables with a
  warning and the pipeline is unaffected.

To disable weapon detection, set `weapon.enabled: false` in `config.yaml`.
Full guide: **`docs/WEAPON_DETECTION.md`**.
