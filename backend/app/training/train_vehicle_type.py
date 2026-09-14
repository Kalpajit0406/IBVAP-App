"""
train_vehicle_type.py — train the dedicated vehicle-type classifier
(models/vehicle_type_classifier.pt) for the 8-class taxonomy Car / Pickup
truck / Truck / Jeep / 2-wheeler / Tanker / Van / Auto-rickshaw
(ibvap/vehicle_type.py CLASSES).

This LOCAL script is scaffolding only — it checks a dataset layout, it does
not fetch or assemble one. For an actual end-to-end run with real data (4
Kaggle datasets covering all 8 classes, best-effort on jeep/pickup_truck —
see its own honesty note), use notebooks/train_vehicle_type_colab.ipynb
instead, then drop the downloaded vehicle_type_classifier.pt here:
models/vehicle_type_classifier.pt. Until a labeled dataset exists under
vehicle_type.dataset_root (config.yaml, default E:/Projects/ibvap-datasets)
for THIS script to use, it has nothing to train on and the pipeline runs on
ibvap/vehicle_type.py's coarse COCO fallback instead.

Unlike the detection fine-tunes (train_weapon.py, train_anpr.py), this is an
image-CLASSIFICATION task: Ultralytics expects a plain ImageFolder layout, not
YOLO-txt labels —

    <dataset_root>/vehicle_type/train/<class_name>/*.jpg
    <dataset_root>/vehicle_type/val/<class_name>/*.jpg

`training/build_vehicle_type_dataset.py` can bootstrap candidate crops from
this deployment's own harvested vehicle footage (data/learning/) into a raw,
coarsely-labeled pool for an operator to hand-sort into the 8 classes — real
footage from this deployment, rather than an internet set of unknown
relevance to Indian roads.

    python training/train_vehicle_type.py --dry-run     # check the dataset layout + counts only
    python training/train_vehicle_type.py               # full train
    python training/train_vehicle_type.py --epochs 60 --batch 32 --imgsz 224
    python training/train_vehicle_type.py --src D:\elsewhere\vehicle_type
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
# Ultralytics classification checkpoints, tried in order. This repo names its
# detection/pose bases "yolo26n(.pt/-pose.pt)"; a matching "-cls.pt" is tried
# first, falling back to a known-good public Ultralytics classification
# checkpoint if that asset isn't available in this ultralytics install.
CLS_BASE_CANDIDATES = ["yolo26n-cls.pt", "yolo11n-cls.pt"]

sys.path.insert(0, str(ROOT))
from ibvap.vehicle_type import CLASSES     # noqa: E402


def _cfg() -> dict:
    return yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


def _emit(obj: dict) -> None:
    print("VEHICLE_TYPE_RESULT " + json.dumps(obj), flush=True)


def _dataset_root() -> Path:
    r = (_cfg().get("vehicle_type", {}) or {}).get("dataset_root")
    return Path(r) if r else Path("E:/Projects/ibvap-datasets")


# ── dataset check / assembly ────────────────────────────────────────────────
def check_layout(src: Path) -> dict:
    """Confirm <src>/{train,val}/<class>/*.jpg exists and count images per
    class. Does not copy anything — unlike the detection fine-tunes, there is
    no raw-download format to convert here; the operator (or
    build_vehicle_type_dataset.py) is expected to have produced this layout
    directly."""
    counts = {"train": {}, "val": {}}
    for split in ("train", "val"):
        split_dir = src / split
        if not split_dir.is_dir():
            continue
        for cls in CLASSES:
            n = len(list((split_dir / cls).glob("*.jp*g"))) if (split_dir / cls).is_dir() else 0
            counts[split][cls] = n
    return counts


def write_report(counts: dict) -> None:
    print(f"  dataset layout: {counts}")
    total_train = sum(counts["train"].values())
    total_val = sum(counts["val"].values())
    missing = [c for c in CLASSES if counts["train"].get(c, 0) == 0]
    print(f"  train={total_train}  val={total_val}")
    if missing:
        print(f"  classes with ZERO training images: {', '.join(missing)}")


# ── train / eval / register ────────────────────────────────────────────────
def _resolve_base() -> str:
    from ultralytics import YOLO
    last_err = None
    for name in CLS_BASE_CANDIDATES:
        try:
            YOLO(name)          # triggers the auto-download / local-cache check
            return name
        except Exception as e:                       # noqa: BLE001
            last_err = e
            print(f"  base checkpoint {name!r} unavailable ({e}) — trying next candidate")
    raise SystemExit(
        f"no usable classification base checkpoint among {CLS_BASE_CANDIDATES}: {last_err}")


def train(data_root: Path, params: dict, ts: str) -> Path:
    from ultralytics import YOLO
    base = _resolve_base()
    m = YOLO(base)
    # Classification training takes the dataset ROOT directly (train/<cls>/*,
    # val/<cls>/*), not a data.yaml like the detection fine-tunes.
    m.train(data=str(data_root), epochs=int(params["epochs"]),
            imgsz=int(params["imgsz"]), batch=int(params["batch"]),
            device=0, project=str(ROOT / "runs" / "classify"),
            name=f"vehicle_type_ft_{ts}", exist_ok=True, verbose=False,
            workers=0, cache=False, plots=True)
    save_dir = Path(getattr(m.trainer, "save_dir",
                            ROOT / "runs" / "classify" / f"vehicle_type_ft_{ts}"))
    for cand in (save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"):
        if cand.exists():
            return cand
    raise SystemExit(f"training produced no weights in {save_dir}")


def evaluate(ft_pt: Path, data_root: Path) -> dict:
    from ultralytics import YOLO
    r = YOLO(str(ft_pt)).val(data=str(data_root), split="val", device=0, verbose=False)
    return {"top1": round(float(r.top1), 4), "top5": round(float(r.top5), 4)}


def register(ft_pt: Path, metrics: dict, counts: dict, ts: str) -> dict:
    models = ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)
    stable = models / "vehicle_type_classifier.pt"
    stamped = models / f"vehicle_type_classifier_{ts}.pt"
    shutil.copy2(ft_pt, stamped)
    shutil.copy2(ft_pt, stable)

    reg_path = models / "vehicle_type_registry.json"
    reg = []
    if reg_path.exists():
        try:
            reg = json.loads(reg_path.read_text("utf-8"))
        except ValueError:
            reg = []
    entry = {"name": f"vehicle_type_{ts}", "weights": f"models/{stamped.name}",
             "active": "models/vehicle_type_classifier.pt", "created": time.time(),
             "metrics": metrics, "counts": counts, "classes": CLASSES}
    reg.append(entry)
    tmp = reg_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    tmp.replace(reg_path)

    rep_dir = ROOT / "docs" / "learning_reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"vehicle_type_{ts}.md").write_text(
        f"# Vehicle-type classifier — {ts}\n\n"
        f"Classes: {', '.join(CLASSES)}.\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| top-1 accuracy | {metrics['top1']:.3f} |\n"
        f"| top-5 accuracy | {metrics['top5']:.3f} |\n\n"
        f"Dataset: {counts}\n\n"
        f"Registered → `models/vehicle_type_classifier.pt`. Restart the server "
        f"(`vehicle_type.enabled: true`) to load it — until then, "
        f"ibvap/vehicle_type.py falls back to the coarse COCO mapping.\n",
        encoding="utf-8")
    return entry


# ── entrypoint ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train the IBVAP vehicle-type classifier (scaffolding — "
                    "needs a hand-labeled dataset, see module docstring)")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the dataset layout + print counts, do not train")
    ap.add_argument("--src", default=None,
                    help="dataset root containing train/<class>/ + val/<class>/ "
                         "(default: <vehicle_type.dataset_root>/vehicle_type)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--min-images", type=int, default=50,
                    help="minimum train images PER CLASS before training proceeds")
    args = ap.parse_args()

    data_root = Path(args.src) if args.src else (_dataset_root() / "vehicle_type")
    print(f"Checking dataset layout under {data_root}")
    if not data_root.exists():
        print(f"  {data_root} does not exist.")
        print("  Build it with training/build_vehicle_type_dataset.py (bootstraps "
              "candidate crops from this deployment's own footage for you to sort), "
              "or place your own train/<class>/*.jpg + val/<class>/*.jpg there.")
        _emit({"state": "failed", "reason": "no_dataset", "path": str(data_root)})
        return 1

    counts = check_layout(data_root)
    write_report(counts)
    thin = [c for c in CLASSES if counts["train"].get(c, 0) < args.min_images]
    if thin:
        print(f"  classes below --min-images {args.min_images}: {', '.join(thin)}")

    if args.dry_run:
        print(f"--dry-run: layout checked at {data_root}. Not training.")
        _emit({"state": "done", "dry_run": True, **counts})
        return 0

    if thin:
        print("Refusing to train with under-represented classes above. "
              "Collect more images or lower --min-images to proceed anyway.")
        _emit({"state": "failed", "reason": "too_few_images", "thin_classes": thin, **counts})
        return 1

    params = {"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz}
    ts = time.strftime("%Y%m%d_%H%M%S")
    print(f"Fine-tuning a classification head  epochs={args.epochs} "
          f"batch={args.batch} imgsz={args.imgsz} …")
    ft_pt = train(data_root, params, ts)
    print(f"  best weights: {ft_pt}")
    print("Evaluating on the held-out val split …")
    metrics = evaluate(ft_pt, data_root)
    print(f"  top1 {metrics['top1']:.3f}  top5 {metrics['top5']:.3f}")
    entry = register(ft_pt, metrics, counts, ts)
    print(f"Registered → models/vehicle_type_classifier.pt  ({entry['name']})")
    print("Restart the server (vehicle_type.enabled: true) to load it.")
    _emit({"state": "done", "name": entry["name"], "metrics": metrics, **counts})
    return 0


if __name__ == "__main__":
    sys.exit(main())
