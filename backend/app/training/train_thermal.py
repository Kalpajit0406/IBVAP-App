"""
train_thermal.py — fine-tune yolo26n for night / thermal (IR) cameras.

Border cameras run infrared at night, where the COCO-trained yolo26n loses most
of its recall. This assembles thermal person + vehicle imagery from LLVIP
(fixed night surveillance IR cams, person) and — if present — FLIR ADAS (thermal
road scenes, vehicles), fine-tunes `yolo26n.pt`, measures base-vs-fine-tuned mAP
on a held-out thermal split, and registers the result in `models/registry.json`
so the dashboard MODEL row can hot-swap to it for IR cameras and back to Nano for
daytime — no restart, no config edit.

    python training/datasets_fetch.py --thermal          # get the data first (into E:/Projects)
    python training/train_thermal.py --dry-run           # assemble + per-class counts, no training
    python training/train_thermal.py                     # fine-tune (freeze=10, 40 epochs)
    python training/train_thermal.py --freeze 0 --epochs 60   # fuller adaptation (slower, better IR)
    python training/train_thermal.py --max-per-source 6000    # cap each source (faster runs)

Classes are remapped to COCO ids and data.yaml is written nc:80 (full COCO
names), exactly like retrain.py — so the fine-tuned head is drop-in for
Detector._classes = [0,2,3,5,7] and the hot-swap needs no server changes.
Colab: notebooks/train_thermal_colab.ipynb.

LLVIP licence is non-commercial research use; FLIR ADAS has its own terms —
see docs/THERMAL_NIGHT.md.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
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
POSE_PT = "yolo26n-pose.pt"
EVAL_CLASSES = [0, 2, 3, 5, 7]

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# source class name -> COCO id (everything else is dropped)
NAME2COCO = {
    "person": 0, "people": 0, "pedestrian": 0,
    "bike": 1, "bicycle": 1,
    "car": 2,
    "motor": 3, "motorcycle": 3, "motorbike": 3,
    "bus": 5,
    "truck": 7,
}


def _cfg() -> dict:
    return yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


def _emit(obj: dict) -> None:
    print("THERMAL_RESULT " + json.dumps(obj), flush=True)


def _dataset_root() -> Path:
    r = (_cfg().get("weapon", {}) or {}).get("dataset_root")
    return Path(r) if r else Path("E:/Projects/ibvap-datasets")


# ── label conversion ──────────────────────────────────────────────────────
def _voc_to_yolo(xml_path: Path, img_wh: tuple | None = None) -> list[str]:
    """VOC XML -> YOLO lines, class names remapped to COCO ids."""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return []
    w = h = 0
    size = root.find("size")
    if size is not None:
        w = int(float(size.findtext("width") or 0))
        h = int(float(size.findtext("height") or 0))
    if (not w or not h) and img_wh:
        w, h = img_wh
    if not w or not h:
        return []
    out = []
    for obj in root.findall("object"):
        cid = NAME2COCO.get((obj.findtext("name") or "").strip().lower())
        if cid is None:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            continue
        out.append(f"{cid} {min(1,max(0,cx)):.6f} {min(1,max(0,cy)):.6f} "
                   f"{min(1,bw):.6f} {min(1,bh):.6f}")
    return out


def _remap_yolo_lines(txt_path: Path, idx2coco: dict) -> list[str]:
    """Remap a YOLO label file's class indices via idx2coco (drop unmapped)."""
    out = []
    for ln in txt_path.read_text("utf-8", "ignore").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        cid = idx2coco.get(int(float(p[0])))
        if cid is None:
            continue
        out.append(f"{cid} {p[1]} {p[2]} {p[3]} {p[4]}")
    return out


# ── source discovery ──────────────────────────────────────────────────────
def _find_llvip(src: Path):
    """(infrared_dir, annotations_dir) or (None, None)."""
    inf = next((p for p in src.rglob("infrared") if p.is_dir()), None)
    ann = next((p for p in src.rglob("Annotations") if p.is_dir()), None)
    return inf, ann


def _find_flir_yolo(src: Path):
    """(data.yaml path, idx2coco) for a Roboflow-style YOLO export, or (None, {})."""
    for dy in src.rglob("data.yaml"):
        if not any((dy.parent / d).exists()
                   for d in ("train", "valid", "test", "labels", "images")):
            continue
        try:
            names = yaml.safe_load(dy.read_text("utf-8")).get("names", [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            idx2coco = {i: NAME2COCO.get(str(n).strip().lower())
                        for i, n in enumerate(names)}
            idx2coco = {i: c for i, c in idx2coco.items() if c is not None}
            if idx2coco:
                return dy, idx2coco
        except Exception:
            continue
    return None, {}


def _find_flir_coco(src: Path) -> list[Path]:
    """FLIR ADAS Kaggle COCO annotation json files."""
    out = []
    for p in src.rglob("*.json"):
        try:
            if p.stat().st_size < 1000:
                continue
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                if '"annotations"' in f.read(4000):
                    out.append(p)
        except OSError:
            continue
    return out


# ── assembly ──────────────────────────────────────────────────────────────
def _take(items: list, cap: int, rng: random.Random) -> list:
    if cap and len(items) > cap:
        rng.shuffle(items)
        return items[:cap]
    return items


def assemble(src_root: Path, out: Path, max_per_source: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    counts = {"llvip_train": 0, "llvip_val": 0, "flir_train": 0, "flir_val": 0,
              "per_class": {}, "no_label": 0}

    def _write(img: Path, lines: list[str], split: str, tag: str) -> None:
        name = f"{tag}_{img.stem}{img.suffix.lower()}"
        shutil.copy2(img, out / "images" / split / name)
        (out / "labels" / split / f"{tag}_{img.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[f"{tag}_{split}"] += 1
        for ln in lines:
            c = int(ln.split()[0])
            counts["per_class"][c] = counts["per_class"].get(c, 0) + 1

    # ---- LLVIP (VOC XML, infrared, person) ----
    inf, ann = _find_llvip(src_root / "llvip")
    if inf and ann:
        imgs = sorted(inf.rglob("*.jpg"))
        folder_split = any("/train/" in p.as_posix() or "/test/" in p.as_posix() for p in imgs)
        tr, va = [], []
        for img in imgs:
            if folder_split:
                split = "val" if "/test/" in img.as_posix() else "train"
            else:
                split = "val" if img.stem.startswith("19") else "train"
            (va if split == "val" else tr).append(img)
        tr = _take(tr, max_per_source, rng)
        va = _take(va, max(300, max_per_source // 8), rng)
        for split, lst in (("train", tr), ("val", va)):
            for img in lst:
                xml = ann / f"{img.stem}.xml"
                lines = _voc_to_yolo(xml) if xml.exists() else []
                if not lines:
                    counts["no_label"] += 1
                _write(img, lines, split, "llvip")
    else:
        print("  LLVIP not found (run: python training/datasets_fetch.py --thermal)")

    # ---- FLIR ADAS (optional) ----
    flir_root = src_root / "flir"
    dy, idx2coco = _find_flir_yolo(flir_root) if flir_root.exists() else (None, {})
    if dy is not None and idx2coco:
        base = dy.parent
        for split_dir, split in (("train", "train"), ("valid", "val"),
                                 ("val", "val"), ("test", "val")):
            lbl_dir = base / split_dir / "labels"
            img_dir = base / split_dir / "images"
            if not (lbl_dir.exists() and img_dir.exists()):
                continue
            pairs = [(p, img_dir / (p.stem + ".jpg")) for p in lbl_dir.glob("*.txt")]
            pairs = [(l, i) for l, i in pairs if i.exists()]
            pairs = _take(pairs, max_per_source if split == "train" else max(300, max_per_source // 8), rng)
            for lbl, img in pairs:
                _write(img, _remap_yolo_lines(lbl, idx2coco), split, "flir")
    elif flir_root.exists() and _find_flir_coco(flir_root):
        _assemble_flir_coco(flir_root, out, max_per_source, rng, counts, _write)
    elif flir_root.exists():
        print(f"  FLIR dir {flir_root} present but no YOLO/COCO layout recognised — skipped")

    n_train = counts["llvip_train"] + counts["flir_train"]
    n_val = counts["llvip_val"] + counts["flir_val"]
    counts["n_train"], counts["n_val"] = n_train, n_val
    return counts


def _assemble_flir_coco(flir_root: Path, out: Path, cap: int, rng: random.Random,
                        counts: dict, _write) -> None:
    for jp in _find_flir_coco(flir_root):
        try:
            data = json.loads(jp.read_text("utf-8"))
        except ValueError:
            continue
        cats = {c["id"]: str(c["name"]).strip().lower() for c in data.get("categories", [])}
        imgs = {im["id"]: im for im in data.get("images", [])}
        by_img: dict = {}
        for a in data.get("annotations", []):
            cid = NAME2COCO.get(cats.get(a.get("category_id"), ""))
            if cid is None or a.get("image_id") not in imgs:
                continue
            by_img.setdefault(a["image_id"], []).append((cid, a["bbox"]))
        split = "val" if "val" in jp.stem.lower() or "test" in jp.stem.lower() else "train"
        items = _take(list(by_img.items()), cap if split == "train" else max(300, cap // 8), rng)
        for img_id, anns in items:
            im = imgs[img_id]
            w, h = im.get("width", 0), im.get("height", 0)
            fp = next((p for p in jp.parent.rglob(Path(im["file_name"]).name)), None)
            if not fp or not w or not h:
                continue
            lines = []
            for cid, (x, y, bw, bh) in anns:
                lines.append(f"{cid} {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
            _write(fp, lines, split, "flir")


def write_data_yaml(out: Path) -> Path:
    y = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
         "nc": 80, "names": COCO_NAMES}
    p = out / "data.yaml"
    p.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return p


# ── train / eval / register ───────────────────────────────────────────────
def train(data_yaml: Path, params: dict, ts: str) -> Path:
    from ultralytics import YOLO
    if not BASE_PT.exists():
        raise SystemExit(f"{BASE_PT} not found — the fine-tune base must be a .pt.")
    m = YOLO(str(BASE_PT))
    m.train(data=str(data_yaml), epochs=int(params["epochs"]),
            imgsz=int(params["imgsz"]), batch=int(params["batch"]),
            freeze=int(params["freeze"]), device=0,
            project=str(ROOT / "runs" / "detect"), name=f"thermal_ft_{ts}",
            exist_ok=True, verbose=False, workers=0, cache=False, plots=True)
    save_dir = Path(getattr(m.trainer, "save_dir",
                            ROOT / "runs" / "detect" / f"thermal_ft_{ts}"))
    for cand in (save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"):
        if cand.exists():
            return cand
    raise SystemExit(f"training produced no weights in {save_dir}")


def evaluate(base_pt: Path, ft_pt: Path, data_yaml: Path) -> dict:
    from ultralytics import YOLO

    def _m(w):
        r = YOLO(str(w)).val(data=str(data_yaml), split="val",
                             classes=EVAL_CLASSES, device=0, verbose=False)
        return {"mAP50": round(float(r.box.map50), 4),
                "mAP50_95": round(float(r.box.map), 4),
                "precision": round(float(r.box.mp), 4),
                "recall": round(float(r.box.mr), 4)}
    base, ft = _m(base_pt), _m(ft_pt)
    return {"base": base, "ft": ft,
            "delta": round(ft["mAP50_95"] - base["mAP50_95"], 4)}


def register(ft_pt: Path, metrics: dict, counts: dict, ts: str) -> dict:
    reg_path = ROOT / _cfg().get("learning", {}).get("paths", {}).get(
        "registry", "models/registry.json")
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    dst = ROOT / "models" / f"yolo26n_thermal_{ts}.pt"
    shutil.copy2(ft_pt, dst)
    entry = {"name": f"thermal_{ts}", "weights": f"models/{dst.name}",
             "pose_weights": POSE_PT, "created": time.time(),
             "label": f"Thermal / night {ts}", "base_mAP": metrics["base"],
             "ft_mAP": metrics["ft"], "delta": metrics["delta"],
             "n_images": counts.get("n_train", 0) + counts.get("n_val", 0),
             "kind": "thermal", "dataset": "LLVIP (+ FLIR ADAS)"}
    reg = []
    if reg_path.exists():
        try:
            reg = json.loads(reg_path.read_text("utf-8"))
        except ValueError:
            reg = []
    reg.append(entry)
    tmp = reg_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    tmp.replace(reg_path)

    rep = ROOT / "docs" / "learning_reports"
    rep.mkdir(parents=True, exist_ok=True)
    pc = counts.get("per_class", {})
    (rep / f"thermal_{ts}.md").write_text(
        f"# Night / thermal fine-tune — {ts}\n\n"
        f"Base `yolo26n.pt`, dataset LLVIP (+ FLIR ADAS). Classes remapped to "
        f"COCO ids, `nc: 80` (drop-in hot-swap).\n\n"
        f"| split | LLVIP | FLIR |\n|---|---|---|\n"
        f"| train | {counts.get('llvip_train',0)} | {counts.get('flir_train',0)} |\n"
        f"| val | {counts.get('llvip_val',0)} | {counts.get('flir_val',0)} |\n\n"
        f"Boxes per COCO id: {pc}\n\n"
        f"| metric | stock yolo26n | thermal ft |\n|---|---|---|\n"
        f"| mAP@50 | {metrics['base']['mAP50']} | {metrics['ft']['mAP50']} |\n"
        f"| mAP@50-95 | {metrics['base']['mAP50_95']} | {metrics['ft']['mAP50_95']} |\n"
        f"| precision | {metrics['base']['precision']} | {metrics['ft']['precision']} |\n"
        f"| recall | {metrics['base']['recall']} | {metrics['ft']['recall']} |\n\n"
        f"Registered as `{entry['name']}` in `models/registry.json` "
        f"(Δ mAP50-95 {metrics['delta']:+.4f}). Restart the server; the dashboard "
        f"MODEL row shows a **{entry['label']}** button. Switch to it for IR "
        f"cameras, back to **Nano** for daytime. See docs/THERMAL_NIGHT.md.\n\n"
        f"Licences: LLVIP is non-commercial research use; FLIR ADAS has its own "
        f"terms — keep this in mind for any commercial deployment.\n",
        encoding="utf-8")
    return entry


# ── entrypoint ────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune yolo26n for night/thermal cameras")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the dataset + print counts, do not train")
    ap.add_argument("--src", default=None,
                    help="dataset root (default: weapon.dataset_root in config.yaml)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--freeze", type=int, default=10,
                    help="freeze the first N layers (0 = full fine-tune, better IR, slower)")
    ap.add_argument("--max-per-source", type=int, default=10000,
                    help="cap train images per source (LLVIP / FLIR) for faster runs")
    ap.add_argument("--min-images", type=int, default=800)
    args = ap.parse_args()

    src_root = Path(args.src) if args.src else _dataset_root()
    out = src_root / "thermal_yolo"
    ts = time.strftime("%Y%m%d_%H%M%S")

    print(f"Assembling {src_root}  →  {out}")
    counts = assemble(src_root, out, args.max_per_source)
    print(f"  train={counts['n_train']}  (LLVIP {counts['llvip_train']} + FLIR {counts['flir_train']})")
    print(f"  val  ={counts['n_val']}  (LLVIP {counts['llvip_val']} + FLIR {counts['flir_val']})")
    print(f"  boxes per COCO id: {counts['per_class']}   frames w/o any label: {counts['no_label']}")

    if counts["n_train"] < args.min_images:
        print(f"Only {counts['n_train']} training images (< --min-images {args.min_images}). "
              f"Run: python training/datasets_fetch.py --thermal")
        _emit({"state": "failed", "reason": "too_few_images", **{k: counts[k] for k in
               ("n_train", "n_val", "llvip_train", "flir_train")}})
        return 1

    data_yaml = write_data_yaml(out)
    if args.dry_run:
        print(f"--dry-run: dataset ready at {data_yaml}. Not training.")
        _emit({"state": "done", "dry_run": True, "n_train": counts["n_train"],
               "n_val": counts["n_val"]})
        return 0

    params = {"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
              "freeze": args.freeze}
    print(f"Fine-tuning {BASE_PT.name}  epochs={args.epochs} batch={args.batch} "
          f"imgsz={args.imgsz} freeze={args.freeze} …")
    ft_pt = train(data_yaml, params, ts)
    print(f"  best weights: {ft_pt}")
    print("Evaluating stock yolo26n vs the thermal fine-tune on the held-out split …")
    metrics = evaluate(BASE_PT, ft_pt, data_yaml)
    print(f"  mAP50-95  stock {metrics['base']['mAP50_95']}  →  ft {metrics['ft']['mAP50_95']}  "
          f"(Δ {metrics['delta']:+.4f})")
    entry = register(ft_pt, metrics, counts, ts)
    print(f"Registered → models/registry.json  ('{entry['name']}', {entry['label']})")
    print("Restart the server; click the new button in the dashboard MODEL row.")
    _emit({"state": "done", "name": entry["name"], "base_mAP": metrics["base"],
           "ft_mAP": metrics["ft"], "delta": metrics["delta"],
           "n_images": entry["n_images"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
