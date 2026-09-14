# Weapon detection (trained firearm object model)

Until now the only "weapon" signal in IBVAP was `ibvap/posture.py` Rule 5
(`chest_aim` / "AIM") — a 2-D skeleton heuristic that cannot see a weapon and
cannot tell a rifle from a broomstick. This adds an actual **trained firearm
detector** and fuses the two.

- **Model:** `models/weapon_detector.pt` — `yolo26n.pt` fine-tuned to one class,
  `gun`. The shipped weights were trained on Colab (T4, 60 epochs): **precision
  0.87 · recall 0.65 · mAP@50 0.75 · mAP@50-95 0.58** on the held-out val split
  (`docs/learning_reports/weapon_20260909_181030.md`,
  `models/weapon_registry.json`). Retrain with `training/train_weapon.py` any time.
- **Data:** [YouTube-GDD](https://github.com/UCAS-GYX/YouTube-GDD) — 5000 images
  from 343 YouTube videos, 16 064 gun instances, YOLO labels, classes person(0) /
  gun(1). It is *dynamic*: guns being drawn, held, aimed and fired, with the
  gun's shape changing through the motion — i.e. it is a held-gun ("gun posture")
  set, not catalogue photos.
- **Runtime:** `ibvap/weapon.py` `WeaponDetector` runs one extra batched
  `predict()` over every camera frame — the same shape as the pose / ANPR
  passes. It never touches the person/vehicle detector or its mAP.

## How a detection becomes an alert (the calibration)

A raw gun box from the model is **not** trusted on its own. `WeaponDetector`
promotes it to `confirmed` only when:

1. `conf >= weapon.min_conf` (default **0.45** — a gun false-positive is a false
   Critical, so this is deliberately high),
2. box area `>= weapon.min_box_area` (**400 px²** — ignore logos / HUD text),
3. it is **held** — the gun box overlaps a tracked person box
   (`IoU >= weapon.person_iou_min`) or its centre sits inside that person's box
   grown 15 %, and
4. the same person track has carried a held gun for `weapon.hold_frames`
   consecutive weapon passes (**3**) — a temporal vote, exactly like AIM's
   `aim_hold_frames`.

A hit that fails 3 or 4 is still returned (`confirmed=False`), drawn faint as
`gun?`, and **never alerts**.

### Fusion tiers (`ibvap/risk_engine.py`)

`RiskAssessment` gains `armed` and `weapon_tier`. On the **same person track**:

| signal | level | score | banner |
|---|---|---|---|
| AIM posture only | Critical | ≥ 92 | `WEAPON (posture)` |
| confirmed gun only | Critical | ≥ 94 | `GUN DETECTED` |
| AIM posture **and** confirmed gun | Critical | 99 | `ARMED THREAT` |

Every first-time `confirmed` gun writes a `{"type":"weapon"}` row to the
SHA-256 hash-chained evidence log (`data/hash_chain.jsonl`), the same override
shape as a virtual-fence breach. `python main.py --verify-chain` still passes.

Confirmed gun crops are also fed to the continuous-learning harvester
(`Harvester.submit_weapon` → `data/learning/weapon.jsonl` + `weapons/*.jpg`) so an
operator can confirm / discard them and grow the training set.

## Getting the data + training

Downloaded datasets live **outside the repo**, under `weapon.dataset_root`
(default `E:/Projects/ibvap-datasets/`).

```bash
python training/datasets_fetch.py --weapon --dry-run     # print the plan, download nothing
python training/datasets_fetch.py --weapon               # git clone + gdown the image zip
#   if gdown is missing it prints the one manual step (pip install gdown; gdown <id>)
#   --from <dir>   register an already-downloaded copy instead of downloading

python training/train_weapon.py --dry-run                # assemble weapon_yolo/ + data.yaml, print counts
python training/train_weapon.py                          # full fine-tune (~1-2 h on an RTX 3050)
python training/train_weapon.py --epochs 60 --batch 16 --imgsz 640
```

`training/train_weapon.py`:

1. `assemble` — YouTube-GDD ships images (`images/{train,val,test}/`, from the
   Google Drive zip) and YOLO labels separately (`labels_only_gun.zip` +
   `YouTube-GDD_test_labels.zip`, in the git repo — `training/datasets_fetch.py` unpacks
   them to `labels/{train,val,test}/` and marks `gun` as class 0). `assemble`
   pairs them, keeps only `gun` boxes as class 0, uses the `train`/`val` folders
   (`test` held out — falls back to a filename split if no folders), and keeps
   gun-less frames as hard negatives capped at 30 % of train (this is what lifts
   precision). Re-running wipes `weapon_yolo/` first.
   Verified assembly on the real download: **4000 train (682 negatives) / 387
   val / 500 test held out**.
2. `write_data_yaml` — `nc: 1`, `names: [gun]`, absolute `path:`.
3. `train` — `YOLO("yolo26n.pt").train(..., workers=0, cache=False, plots=True)`.
4. `evaluate` — `model.val(split="val")` → precision / recall / mAP50 / mAP50-95.
5. `register` — copies `best.pt` → `models/weapon_detector.pt` (+ a timestamped
   copy), appends `models/weapon_registry.json`, writes
   `docs/learning_reports/weapon_<ts>.md`.

Restart the server (`weapon.enabled: true`) to pick up the new weights.

### Colab

`notebooks/train_weapon_colab.ipynb` runs the same steps on a free T4 (fetch →
`train_weapon.py --src /content/ytgdd` → download `weapon_detector.pt`). Use it
if the laptop run is too slow.

## Config (`config.yaml` `weapon:`)

| key | meaning |
|---|---|
| `enabled` | master switch. If the weights are missing the engine self-disables and the pipeline is unaffected. |
| `weights` | `models/weapon_detector.pt` |
| `dataset_root` | where `training/datasets_fetch.py` / `training/train_weapon.py` put downloaded data — **outside the repo** |
| `min_conf` | promote a raw box only above this (0.45) |
| `min_box_area` | px² floor (400) |
| `person_iou_min` | gun∧person overlap to count as "held" (0.05) |
| `hold_frames` | consecutive weapon passes before "confirmed" (3) |
| `detect_every` | run the weapon model every Nth detection tick (**3**). It is a PyTorch `.pt`, not the TensorRT engine — on the RTX 3050 with detector+pose+ANPR also running, every-2 cost ~1/3 of worker fps at 2 cams; every-3 holds it. Lower on a bigger GPU / after a TensorRT export. |
| `imgsz` | weapon-model input size (640) |
| `hit_ttl_s` | keep drawing a confirmed box this long between passes (1.5 s) |

## Honest limits

- **The model is only as good as YouTube-GDD.** It is handgun/rifle-heavy from
  YouTube footage — expect misses on unusual angles, small/distant guns, and
  low light. Retrain with more data (Roboflow / operator-confirmed crops from
  the harvester) to improve.
- A **one-handed pistol grip** may not raise the AIM posture tier (that rule
  needs two hands); the object detector covers that case, but only once the gun
  is large enough to detect.
- Association is bounding-box only. A gun held out to the side, past the person
  box + 15 %, won't be linked. Wrist-keypoint association is a future tweak.
- `min_conf` 0.45 trades recall for precision on purpose. Tune it against the
  `docs/learning_reports/weapon_<ts>.md` P/R numbers for your footage.
