# Night / thermal model (`training/train_thermal.py`)

IBVAP's detector is `yolo26n` trained on COCO — daytime RGB. Border cameras run
**infrared at night**, where a COCO model loses most of its recall (no colour,
white-hot blobs, different texture). `training/train_thermal.py` fine-tunes `yolo26n.pt`
on thermal imagery and registers the result as a **hot-swappable model**: the
operator flips the dashboard **MODEL** row to *Thermal / night* for IR cameras
and back to *Nano* for daytime — no restart, no config edit.

It reuses the exact model-registry hot-swap path already built for the
continuous-learning fine-tunes (`_read_registry` → `/api/models` →
`/api/switch-model {weights_ref}` → `_switch_q` dict → `Detector.from_weights`),
so **no server / detector / dashboard code changes** were needed.

**Shipped model** (`models/registry.json` → `thermal_20260909_211551`, trained
on Colab): on the held-out night split it lifts recall **0.53 → 0.72** and
mAP@50-95 **0.365 → 0.525** (+0.16) over stock `yolo26n`. Full numbers +
caveats: `docs/learning_reports/thermal_20260909_211551.md`. The weights
themselves are `.gitignore`d (large, site-specific) — re-run `train_thermal.py`
or the Colab notebook to regenerate.

## Datasets

| dataset | what | how | licence |
|---|---|---|---|
| **LLVIP** ([bupt-ai-cz/LLVIP](https://github.com/bupt-ai-cz/LLVIP)) | 15,488 infrared frames from **fixed night surveillance cameras**, VOC-XML **person** labels | one Google Drive zip via `gdown` (~4 GB), no key | **non-commercial research use** |
| **FLIR ADAS** ([Teledyne](https://www.flir.com/oem/adas/adas-dataset-form/)) | thermal road scenes — adds **car / bus / truck / bicycle / motorcycle** | free **Roboflow** key (YOLO-format mirror, ~2–3 GB) *or* a Kaggle token (v2, COCO JSON, ~15 GB) | Teledyne FLIR dataset terms |

LLVIP is the load-bearing set (it is literally night surveillance IR). FLIR is
optional — without a key `training/train_thermal.py` builds a **LLVIP-only person night
model**, which is still a useful swap.

**Licence note:** LLVIP is released for non-commercial research. IBVAP is an
SIH / MHA prototype, which is fine; flag it before any commercial deployment and
swap in site-collected footage.

## Class mapping — keep COCO ids, `nc: 80`

Every source class is remapped to a COCO id and `data.yaml` is written `nc: 80`
with the full COCO names — same trick as `training/retrain.py`. The fine-tuned head is
then identical in shape to `yolo26n.pt`, `Detector._classes = [0,2,3,5,7]`
filters unchanged, and the hot-swap is genuinely drop-in.

| source name | → COCO id |
|---|---|
| `person` / `people` / `pedestrian` | 0 |
| `bike` / `bicycle` | 1 |
| `car` | 2 |
| `motor` / `motorcycle` / `motorbike` | 3 |
| `bus` | 5 |
| `truck` | 7 |
| anything else (traffic light, dog, sign, …) | dropped |

mAP is measured with `model.val(classes=[0,2,3,5,7])` for both stock `yolo26n`
and the fine-tune — the gap is the dashboard badge.

## Run

```bash
# 1. get the data (into E:/Projects/ibvap-datasets — never the repo)
python training/datasets_fetch.py --thermal                     # LLVIP only
python training/datasets_fetch.py --thermal --roboflow-key KEY  # + FLIR ADAS (vehicles)
python training/datasets_fetch.py --thermal --kaggle            # + FLIR ADAS v2 via ~/.kaggle/kaggle.json

# 2. assemble + fine-tune
python training/train_thermal.py --dry-run                       # counts only, no training
python training/train_thermal.py                                 # freeze=10, 40 epochs, ~1.5-2 h on an RTX 3050
python training/train_thermal.py --freeze 0 --epochs 60          # fuller IR adaptation (slower, better)
python training/train_thermal.py --max-per-source 6000           # cap each source for a quicker run
```

`training/train_thermal.py`:

1. `assemble` — LLVIP `infrared/*.jpg` + `Annotations/<stem>.xml` (VOC→YOLO,
   person→0); split by folder if `infrared/train,test` exist, else by filename
   (`19*` = val, LLVIP's own convention). FLIR: Roboflow YOLO export (remap class
   indices) or Kaggle COCO JSON (parse + convert). `--max-per-source` caps each.
2. `write_data_yaml` — `nc: 80`, full COCO names, absolute `path:`.
3. `train` — `YOLO("yolo26n.pt").train(freeze=10, workers=0, cache=False)`.
4. `evaluate` — `val(split="val", classes=[0,2,3,5,7])` for stock vs fine-tuned.
5. `register` — copies `best.pt` → `models/yolo26n_thermal_<ts>.pt`, appends
   `models/registry.json` (`label: "Thermal / night <ts>"`, `kind: "thermal"`),
   writes `docs/learning_reports/thermal_<ts>.md`.

### Colab

`notebooks/train_thermal_colab.ipynb` is self-contained: `gdown` LLVIP → optional
Roboflow FLIR (paste key) → inline assemble → `train(epochs=30, freeze=10)` →
downloads `yolo26n_thermal.pt` **and** a `registry_entry.json`. Merge it locally:

```bash
python -c "import json,pathlib; p=pathlib.Path('models/registry.json'); r=json.loads(p.read_text()) if p.exists() else []; r.append(json.load(open('registry_entry.json'))); p.write_text(json.dumps(r,indent=2))"
```

## Using it (operator workflow)

1. `python server.py` (any mode). `GET /api/models` now lists `thermal_<ts>`
   with a `+XX.X mAP` badge; the dashboard **MODEL** row shows a **Thermal /
   night** button.
2. When the IR / night cameras come online, click it — the detector hot-swaps
   in ~1–3 s (`Model hot-swapped to 'thermal_<ts>'` in the log). ByteTrack,
   pose, ANPR, weapon, geofence, evidence all keep running.
3. At daybreak click **Nano** to roll back. Evidence chain is untouched
   (`python main.py --verify-chain` → `Chain OK`).

## Honest limits

- **LLVIP is one city, person-only, one camera style.** It teaches "person in
  IR" well but nothing about *your* site's cameras. FLIR ADAS is dashcam-angle,
  not fixed CCTV. Expect a real jump over stock `yolo26n` on thermal, but for
  production, fine-tune again on footage from the actual deployed IR cameras
  (the continuous-learning harvester can collect it once the thermal model is
  live).
- Thermal vehicles depend entirely on whether FLIR was included.
- The model is `yolo26n` size — swap cost and VRAM are the same as Nano, so the
  hot-swap is safe on the 6 GB card.

`tests/test_thermal.py` covers the VOC→YOLO conversion, the class remap, and the
LLVIP split rule (no GPU).
