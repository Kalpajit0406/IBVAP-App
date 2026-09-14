"""
train_weapon.py — train the dedicated firearm detector (models/weapon_detector.pt).

Assembles the YouTube-GDD download (fetched by `datasets_fetch.py` into
`weapon.dataset_root`, i.e. E:/Projects/ibvap-datasets/) into a single-class
`gun` YOLO dataset, fine-tunes `yolo26n.pt` on it, measures precision / recall /
mAP on the held-out fold, and drops `models/weapon_detector.pt` into the repo so
the pipeline picks it up on the next restart (config.yaml `weapon:`).

    python training/train_weapon.py --dry-run     # assemble + print counts, no training
    python training/train_weapon.py               # full train (~1-2 h on an RTX 3050)
    python training/train_weapon.py --epochs 60 --batch 16 --imgsz 640
    python training/train_weapon.py --src D:\elsewhere\youtube-gdd

`datasets_fetch.py` unpacks YouTube-GDD's `labels_only_gun.zip` (gun = class 0)
next to the images. We keep only the `gun` boxes as class 0 — so the trained
head is a clean 1-class `gun` model — and the person(0)/gun(1) raw set is still
handled (auto-detected via `_gun_class`). Gun-less frames are kept as capped
hard negatives, which is what pushes precision up. Split is by the
`images/{train,val,test}/` folders (test held out).

Colab: notebooks/train_weapon_colab.ipynb runs these same steps on a free T4.
"""
from __future__ import annotations

import argparse
import json
import random
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
BASE_PT = ROOT / "yolo26n.pt"
NEG_FRACTION = 0.30           # cap person-only (no-gun) frames at this share of train


def _cfg() -> dict:
    return yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


def _emit(obj: dict) -> None:
    print("WEAPON_RESULT " + json.dumps(obj), flush=True)


def _dataset_root() -> Path:
    r = (_cfg().get("weapon", {}) or {}).get("dataset_root")
    return Path(r) if r else Path("E:/Projects/ibvap-datasets")


# ── dataset assembly ───────────────────────────────────────────────────────
def _find_pairs(src: Path) -> list[tuple[Path, Path | None, str | None]]:
    """[(image, sibling label or None, split from the folder or None)] for every
    image under an `images/` dir. YouTube-GDD ships images/{train,val,test}/ and,
    once datasets_fetch.py has unpacked them, labels/{train,val,test}/."""
    pairs = []
    for img in src.rglob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        parent_names = {p.name.lower() for p in img.parents}
        if "images" not in parent_names:
            continue
        parts = list(img.parts)
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
        split = next((s for s in ("train", "val", "test") if s in parent_names), None)
        pairs.append((img, lbl if lbl.exists() else None, split))
    return pairs


def _gun_class(src: Path) -> int:
    """Which class id is `gun`? 1 in raw YouTube-GDD (person=0, gun=1); 0 in
    the labels_only_gun set.

    The label files decide, not the .gun_class marker: datasets_fetch writes
    the marker even when the full two-class labels end up on disk, and
    trusting it silently turned every PERSON box into a "gun" label (a model
    retrained on that learns to call people guns). A gun-only set never
    contains class 1, so seeing class 1 settles it."""
    seen: set[str] = set()
    for lbl in list(src.rglob("labels/*/*.txt"))[:400]:
        for ln in lbl.read_text("utf-8", "ignore").splitlines():
            q = ln.split()
            if q:
                seen.add(q[0])
    if "1" in seen:
        return 1
    marker = list(src.rglob("labels/.gun_class"))
    if marker:
        try:
            return int(marker[0].read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass
    return 0


def _convert_label(lbl: Path | None, gun_id: int = 1) -> tuple[str, bool]:
    """YouTube-GDD label → single-class `gun` (class 0). Returns (text, has_gun)."""
    if lbl is None or not lbl.exists():
        return "", False
    out = []
    for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        if p[0] == str(gun_id):
            out.append("0 " + " ".join(p[1:]))
    return ("\n".join(out) + ("\n" if out else "")), bool(out)


def assemble(src_root: Path, out: Path, val_fraction: float, seed: int = 0) -> dict:
    src = src_root
    if (src_root / "youtube-gdd").exists():
        src = src_root / "youtube-gdd"
    pairs = _find_pairs(src)
    if not pairs:
        raise SystemExit(
            f"no images found under {src}. Run: python training/datasets_fetch.py --weapon")

    gun_id = _gun_class(src)
    n_lbl = sum(1 for _i, l, _s in pairs if l is not None)
    have_folders = any(s in ("train", "val") for _i, _l, s in pairs)
    print(f"  {len(pairs)} images, {n_lbl} with a label file, gun = class {gun_id}")
    if have_folders:
        print("  split: images/{train,val}/ folders (images/test/ held out)")
    else:
        print(f"  no train/val folders — splitting by sorted filename "
              f"(last {val_fraction:.0%} = val)")

    pairs.sort(key=lambda t: t[0].name)
    n_val_cut = max(1, int(len(pairs) * val_fraction))

    # Start clean so a re-run never accumulates stale images/labels.
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    n = {"train": 0, "val": 0, "train_neg": 0, "val_gun": 0, "skipped_test": 0}
    rng = random.Random(seed)
    for idx, (img, lbl, folder_split) in enumerate(pairs):
        if have_folders:
            if folder_split == "test" or folder_split is None:
                n["skipped_test"] += 1
                continue
            split = folder_split
        else:
            split = "val" if idx >= len(pairs) - n_val_cut else "train"

        text, has_gun = _convert_label(lbl, gun_id)
        if not has_gun:
            if split == "train":
                if n["train_neg"] >= NEG_FRACTION * max(n["train"], 1) + 25:
                    continue
                n["train_neg"] += 1
            # keep a few negatives in val too, but don't inflate it
            elif rng.random() > 0.15:
                continue
        else:
            n[f"{split}_gun"] = n.get(f"{split}_gun", 0) + 1

        dst_img = out / "images" / split / img.name
        shutil.copy2(img, dst_img)
        (out / "labels" / split / (img.stem + ".txt")).write_text(text, encoding="utf-8")
        n[split] += 1

    return {"n_train": n["train"], "n_val": n["val"],
            "n_train_neg": n["train_neg"], "n_val_gun": n.get("val_gun", 0),
            "n_skipped_test": n["skipped_test"], "n_pairs": len(pairs)}


def write_data_yaml(out: Path) -> Path:
    y = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
         "nc": 1, "names": ["gun"]}
    p = out / "data.yaml"
    p.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return p


# ── train / eval / register ────────────────────────────────────────────────
def train(data_yaml: Path, params: dict, ts: str) -> Path:
    from ultralytics import YOLO
    if not BASE_PT.exists():
        raise SystemExit(f"{BASE_PT} not found — the fine-tune base must be a .pt.")
    m = YOLO(str(BASE_PT))
    m.train(data=str(data_yaml), epochs=int(params["epochs"]),
            imgsz=int(params["imgsz"]), batch=int(params["batch"]),
            device=0, project=str(ROOT / "runs" / "detect"),
            name=f"weapon_ft_{ts}", exist_ok=True, verbose=False,
            workers=0, cache=False, plots=True)
    save_dir = Path(getattr(m.trainer, "save_dir",
                            ROOT / "runs" / "detect" / f"weapon_ft_{ts}"))
    for cand in (save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"):
        if cand.exists():
            return cand
    raise SystemExit(f"training produced no weights in {save_dir}")


def evaluate(ft_pt: Path, data_yaml: Path) -> dict:
    from ultralytics import YOLO
    r = YOLO(str(ft_pt)).val(data=str(data_yaml), split="val", device=0, verbose=False)
    b = r.box
    return {"precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4),
            "mAP50": round(float(b.map50), 4), "mAP50_95": round(float(b.map), 4)}


def register(ft_pt: Path, metrics: dict, counts: dict, ts: str) -> dict:
    models = ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)
    stable = models / "weapon_detector.pt"
    stamped = models / f"weapon_detector_{ts}.pt"
    shutil.copy2(ft_pt, stamped)
    shutil.copy2(ft_pt, stable)

    reg_path = models / "weapon_registry.json"
    reg = []
    if reg_path.exists():
        try:
            reg = json.loads(reg_path.read_text("utf-8"))
        except ValueError:
            reg = []
    entry = {"name": f"weapon_{ts}", "weights": f"models/{stamped.name}",
             "active": "models/weapon_detector.pt", "created": time.time(),
             "metrics": metrics, "counts": counts, "base": "yolo26n.pt",
             "dataset": "YouTube-GDD (github.com/UCAS-GYX/YouTube-GDD)"}
    reg.append(entry)
    tmp = reg_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    tmp.replace(reg_path)

    rep_dir = ROOT / "docs" / "learning_reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"weapon_{ts}.md").write_text(
        f"# Weapon detector — {ts}\n\n"
        f"Base `yolo26n.pt`, dataset YouTube-GDD (gun class only).\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| precision | {metrics['precision']:.3f} |\n"
        f"| recall | {metrics['recall']:.3f} |\n"
        f"| mAP@50 | {metrics['mAP50']:.3f} |\n"
        f"| mAP@50-95 | {metrics['mAP50_95']:.3f} |\n\n"
        f"Dataset: {counts}\n\n"
        f"Registered → `models/weapon_detector.pt`. Restart the server "
        f"(`weapon.enabled: true`) to load it.\n\n"
        f"Tuning: raise `weapon.min_conf` if precision is low (false Criticals), "
        f"lower it / raise `hold_frames` if recall is low. See docs/WEAPON_DETECTION.md.\n",
        encoding="utf-8")
    return entry


# ── entrypoint ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Train the IBVAP firearm detector")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the dataset + print counts, do not train")
    ap.add_argument("--src", default=None,
                    help="dataset root (default: weapon.dataset_root in config.yaml)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--min-images", type=int, default=200)
    args = ap.parse_args()

    src_root = Path(args.src) if args.src else _dataset_root()
    out = src_root / "weapon_yolo"
    ts = time.strftime("%Y%m%d_%H%M%S")

    print(f"Assembling {src_root}  →  {out}")
    counts = assemble(src_root, out, args.val_fraction)
    print(f"  train={counts['n_train']} (neg {counts['n_train_neg']})  "
          f"val={counts['n_val']} (gun {counts['n_val_gun']})  "
          f"held-out(test)={counts['n_skipped_test']}  of {counts['n_pairs']} pairs")

    if counts["n_train"] < args.min_images:
        print(f"Only {counts['n_train']} training images (< --min-images "
              f"{args.min_images}). Fetch the full dataset: python training/datasets_fetch.py --weapon")
        _emit({"state": "failed", "reason": "too_few_images", **counts})
        return 1

    data_yaml = write_data_yaml(out)
    if args.dry_run:
        print(f"--dry-run: dataset ready at {data_yaml}. Not training.")
        _emit({"state": "done", "dry_run": True, **counts})
        return 0

    params = {"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz}
    print(f"Fine-tuning {BASE_PT.name}  epochs={args.epochs} batch={args.batch} "
          f"imgsz={args.imgsz} …  (this is the ~1-2 h step)")
    ft_pt = train(data_yaml, params, ts)
    print(f"  best weights: {ft_pt}")
    print("Evaluating on the held-out val split …")
    metrics = evaluate(ft_pt, data_yaml)
    print(f"  precision {metrics['precision']:.3f}  recall {metrics['recall']:.3f}  "
          f"mAP50 {metrics['mAP50']:.3f}  mAP50-95 {metrics['mAP50_95']:.3f}")
    entry = register(ft_pt, metrics, counts, ts)
    print(f"Registered → models/weapon_detector.pt  ({entry['name']})")
    print("Restart the server (weapon.enabled: true) to load it.")
    _emit({"state": "done", "name": entry["name"], "metrics": metrics, **counts})
    return 0


if __name__ == "__main__":
    sys.exit(main())
