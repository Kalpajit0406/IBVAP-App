"""
build_vehicle_type_dataset.py — bootstrap a starting point for the vehicle-type
dataset (see training/train_vehicle_type.py) from THIS deployment's own
harvested footage, rather than an internet set of unknown relevance to Indian
roads.

The continuous-learning harvester (ibvap/learn.py) already saves
data/learning/frames/<id>.jpg + data/learning/labels/<id>.txt (YOLO-format,
raw COCO class ids — see ibvap.learn.remap_class, identity today). This script
crops every vehicle box (COCO car/motorcycle/bus/truck = 2/3/5/7) out of that
pool and sorts each crop into a COARSE bucket — car / two_wheeler / van / truck
— using the same mapping ibvap/vehicle_type.py falls back to
(_COCO_FALLBACK). That's only 4 of the 8 target classes: the operator still
has to manually split each coarse bucket further (e.g. some "van" crops are
really tankers, some "car" crops are really jeeps) and pull out auto-rickshaws
by hand — COCO has no class for jeep / pickup truck / tanker / auto-rickshaw
at all, so no automated pass can place a crop directly into those buckets.

Output layout (an unsorted POOL, not yet the train/val split
train_vehicle_type.py expects — moving files between these folders, or further
splitting one, is a manual step done by a person before training):

    <vehicle_type.dataset_root>/vehicle_type_raw/<coarse_label>/<frame_id>_<n>.jpg

    python training/build_vehicle_type_dataset.py --dry-run   # count only, write nothing
    python training/build_vehicle_type_dataset.py             # crop + write
    python training/build_vehicle_type_dataset.py --min-box-area 900
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ibvap.vehicle_type import _COCO_FALLBACK     # noqa: E402

_VEHICLE_IDS = set(_COCO_FALLBACK)     # {2, 3, 5, 7}


def _cfg() -> dict:
    return yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


def _dataset_root() -> Path:
    r = (_cfg().get("vehicle_type", {}) or {}).get("dataset_root")
    return Path(r) if r else Path("E:/Projects/ibvap-datasets")


def _learning_root() -> Path:
    r = (_cfg().get("learning", {}) or {}).get("dir")
    return Path(r) if r else Path("data/learning")


def _iter_vehicle_boxes(labels_dir: Path):
    """(frame_id, coco_class_id, cx, cy, bw, bh) for every vehicle line in
    every label file — cx/cy/bw/bh normalised 0..1, matching
    ibvap.learn.to_yolo_line."""
    for lbl in sorted(labels_dir.glob("*.txt")):
        try:
            lines = lbl.read_text("utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, bw, bh = (float(v) for v in parts[1:])
            except ValueError:
                continue
            if cid in _VEHICLE_IDS:
                yield lbl.stem, cid, cx, cy, bw, bh


def build(dry_run: bool, min_box_area: int) -> dict:
    learn_root = _learning_root()
    frames_dir, labels_dir = learn_root / "frames", learn_root / "labels"
    if not labels_dir.is_dir():
        print(f"{labels_dir} not found — nothing harvested yet "
              f"(learning.enabled must have been on for a while first).")
        return {"n_boxes": 0, "n_written": 0, "by_label": {}}

    out_root = _dataset_root() / "vehicle_type_raw"
    by_label: dict[str, int] = {c: 0 for c in set(_COCO_FALLBACK.values())}
    n_boxes = n_written = n_skipped_small = n_missing_frame = 0
    t0 = time.time()

    for i, (frame_id, cid, cx, cy, bw, bh) in enumerate(_iter_vehicle_boxes(labels_dir)):
        n_boxes += 1
        if i and i % 500 == 0:
            print(f"  ...{i} vehicle boxes scanned, {n_written} crops written "
                  f"({time.time() - t0:.0f}s)")
        frame_path = frames_dir / f"{frame_id}.jpg"
        if not frame_path.is_file():
            n_missing_frame += 1
            continue
        label = _COCO_FALLBACK[cid]
        if dry_run:
            by_label[label] += 1
            n_written += 1
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            n_missing_frame += 1
            continue
        h, w = img.shape[:2]
        x1 = max(0, int((cx - bw / 2) * w)); y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w)); y2 = min(h, int((cy + bh / 2) * h))
        if (x2 - x1) * (y2 - y1) < min_box_area:
            n_skipped_small += 1
            continue
        crop = img[y1:y2, x1:x2]
        dest_dir = out_root / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        n = by_label[label]
        cv2.imwrite(str(dest_dir / f"{frame_id}_{n}.jpg"), crop)
        by_label[label] += 1
        n_written += 1

    return {"n_boxes": n_boxes, "n_written": n_written, "by_label": by_label,
            "n_skipped_small": n_skipped_small, "n_missing_frame": n_missing_frame,
            "out_root": str(out_root)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bootstrap a coarse-labeled vehicle-crop pool from this "
                    "deployment's own harvested footage (data/learning/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="count what would be written, write nothing")
    ap.add_argument("--min-box-area", type=int, default=900,
                    help="px\u00b2 \u2014 skip crops smaller than this (default 30x30)")
    args = ap.parse_args()

    print("Scanning data/learning/ for vehicle boxes...")
    r = build(args.dry_run, args.min_box_area)
    print(f"\n{r['n_boxes']} vehicle box(es) found, "
          f"{r['n_written']} crop(s) {'would be ' if args.dry_run else ''}written")
    if r.get("by_label"):
        for label, n in sorted(r["by_label"].items()):
            print(f"  {label:14s} {n}")
    if r.get("n_skipped_small"):
        print(f"  ({r['n_skipped_small']} skipped as too small)")
    if r.get("n_missing_frame"):
        print(f"  ({r['n_missing_frame']} label file(s) with no matching frame)")
    if not args.dry_run and r.get("n_written"):
        print(f"\nWritten to {r['out_root']}. This is only a COARSE 4-bucket "
              f"starting pool (car/two_wheeler/van/truck) — hand-sort it into "
              f"train_vehicle_type.py's train/<class>/ + val/<class>/ layout, "
              f"splitting jeep out of car, pickup_truck/tanker out of "
              f"truck/van, and pulling auto_rickshaw out by hand (no COCO "
              f"class maps to it automatically).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
