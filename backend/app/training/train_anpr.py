"""
train_anpr.py — fix ANPR: fine-tune the plate DETECTOR and prepare the plate OCR.

IBVAP's ANPR is two stages and both are weak on Indian plates:

  --detector   fine-tune yolo26n on Indian licence-plate datasets  ->
               models/license_plate_detector.pt  (config anpr.weights points here)
  --ocr        assemble a fast-plate-ocr training set (Kaggle/Roboflow crops +
               operator-confirmed crops from data/learning/plates/) and, if
               `fast-plate-ocr[train]` is installed, fine-tune
               cct-s-v2-global-model -> models/plate_ocr/. Otherwise it writes
               the dataset + the next command; do the fine-tune on Kaggle
               (notebooks/train_anpr_kaggle.ipynb). The pretrained
               cct-s-v2-global-model already beats generic EasyOCR.

    python training/datasets_fetch.py --anpr        # get the data first (into E:/Projects)
    python training/train_anpr.py --detector --dry-run
    python training/train_anpr.py --detector
    python training/train_anpr.py --ocr --dry-run
    python training/train_anpr.py --ocr

Datasets live under weapon.dataset_root (E:/Projects/ibvap-datasets/anpr/), never
the repo. Trained artifacts: models/license_plate_detector.pt (shipped) and
models/plate_ocr/ (shipped) + models/anpr_registry.json (tracked).
"""
from __future__ import annotations

import argparse
import csv
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
PLATE_NAMES = {"license_plate", "licence_plate", "licence-plate", "license-plate",
               "number_plate", "numberplate", "plate", "np", "lp", "0"}
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _cfg() -> dict:
    return yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


def _emit(obj: dict) -> None:
    print("ANPR_RESULT " + json.dumps(obj), flush=True)


def _dataset_root() -> Path:
    r = (_cfg().get("weapon", {}) or {}).get("dataset_root")
    return Path(r) if r else Path("E:/Projects/ibvap-datasets")


# ── detector: assemble a single-class `license_plate` YOLO set ──────────────
def _voc_plate_lines(xml_path: Path) -> list[str]:
    """Every <object> in a VOC XML -> `0 cx cy w h` (single-class plate)."""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return []
    s = root.find("size")
    w = int(float(s.findtext("width") or 0)) if s is not None else 0
    h = int(float(s.findtext("height") or 0)) if s is not None else 0
    if not w or not h:
        return []
    out = []
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip().lower()
        if name and name not in PLATE_NAMES and not name.startswith(("plate", "lp", "number")):
            continue
        b = o.find("bndbox")
        if b is None:
            continue
        try:
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            continue
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        if bw > 0 and bh > 0:
            out.append(f"0 {min(1,max(0,cx)):.6f} {min(1,max(0,cy)):.6f} "
                       f"{min(1,bw):.6f} {min(1,bh):.6f}")
    return out


def _yolo_lines_to_class0(txt: Path) -> list[str]:
    out = []
    for ln in txt.read_text("utf-8", "ignore").splitlines():
        p = ln.split()
        if len(p) == 5:
            out.append("0 " + " ".join(p[1:]))
    return out


def assemble_detector(src_root: Path, out: Path, max_per_source: int,
                      val_fraction: float = 0.1, seed: int = 0) -> dict:
    src = src_root / "anpr"
    if not src.exists():
        raise SystemExit(f"{src} missing — run: python training/datasets_fetch.py --anpr")
    rng = random.Random(seed)
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    # every image that has a usable label (sibling YOLO .txt or a matching .xml)
    items: list[tuple[Path, list[str], str]] = []   # (img, yolo_lines, source_tag)
    per_src: dict[str, int] = {}
    for img in src.rglob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        tag = img.relative_to(src).parts[0] if img.relative_to(src).parts else "misc"
        lines: list[str] = []
        yolo = Path(str(img).replace(f"{img.parent.name}", "labels", 1) if img.parent.name == "images"
                    else str(img)).with_suffix(".txt")
        cand = [img.with_suffix(".txt"),
                img.parent.parent / "labels" / (img.stem + ".txt"),
                Path(str(img.parent).replace("images", "labels")) / (img.stem + ".txt")]
        for c in cand:
            if c.exists():
                lines = _yolo_lines_to_class0(c)
                break
        if not lines:
            xc = [img.with_suffix(".xml"),
                  img.parent.parent / "annotations" / (img.stem + ".xml"),
                  Path(str(img.parent).replace("images", "annotations")) / (img.stem + ".xml")]
            for c in xc:
                if c.exists():
                    lines = _voc_plate_lines(c)
                    break
        if not lines:
            continue
        if per_src.get(tag, 0) >= max_per_source:
            continue
        per_src[tag] = per_src.get(tag, 0) + 1
        items.append((img, lines, tag))

    if not items:
        raise SystemExit(f"no image+label pairs under {src} — check the download layout")
    rng.shuffle(items)
    n_val = max(20, int(len(items) * val_fraction))
    n = {"train": 0, "val": 0}
    for i, (img, lines, tag) in enumerate(items):
        split = "val" if i < n_val else "train"
        name = f"{tag}_{img.stem}{img.suffix.lower()}"
        shutil.copy2(img, out / "images" / split / name)
        (out / "labels" / split / f"{tag}_{img.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        n[split] += 1
    return {"n_train": n["train"], "n_val": n["val"], "per_source": per_src}


def write_data_yaml(out: Path) -> Path:
    y = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
         "nc": 1, "names": ["license_plate"]}
    p = out / "data.yaml"
    p.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return p


def train_detector(data_yaml: Path, params: dict, ts: str) -> Path:
    from ultralytics import YOLO
    if not BASE_PT.exists():
        raise SystemExit(f"{BASE_PT} not found.")
    m = YOLO(str(BASE_PT))
    m.train(data=str(data_yaml), epochs=int(params["epochs"]), imgsz=int(params["imgsz"]),
            batch=int(params["batch"]), device=0, project=str(ROOT / "runs" / "detect"),
            name=f"anpr_det_{ts}", exist_ok=True, verbose=False,
            workers=0, cache=False, plots=True)
    sd = Path(getattr(m.trainer, "save_dir", ROOT / "runs" / "detect" / f"anpr_det_{ts}"))
    for c in (sd / "weights" / "best.pt", sd / "weights" / "last.pt"):
        if c.exists():
            return c
    raise SystemExit(f"no weights in {sd}")


def evaluate_detector(base_pt, ft_pt, data_yaml) -> dict:
    from ultralytics import YOLO

    def _m(w):
        r = YOLO(str(w)).val(data=str(data_yaml), split="val", device=0, verbose=False).box
        return {"mAP50": round(float(r.map50), 4), "mAP50_95": round(float(r.map), 4),
                "precision": round(float(r.mp), 4), "recall": round(float(r.mr), 4)}
    base, ft = _m(base_pt), _m(ft_pt)
    return {"base": base, "ft": ft, "delta": round(ft["mAP50_95"] - base["mAP50_95"], 4)}


def register_detector(ft_pt: Path, metrics: dict, counts: dict, ts: str) -> dict:
    models = ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ft_pt, models / f"license_plate_detector_{ts}.pt")
    shutil.copy2(ft_pt, models / "license_plate_detector.pt")     # replace the live one
    reg_p = models / "anpr_registry.json"
    reg = json.loads(reg_p.read_text("utf-8")) if reg_p.exists() else []
    reg.append({"kind": "detector", "name": f"anpr_det_{ts}",
                "weights": "models/license_plate_detector.pt", "created": time.time(),
                "metrics": metrics, "counts": counts, "base": "yolo26n.pt"})
    reg_p.with_suffix(".json.tmp").write_text(json.dumps(reg, indent=2), encoding="utf-8")
    reg_p.with_suffix(".json.tmp").replace(reg_p)
    _report(f"anpr_det_{ts}", "Plate detector (YOLO)", metrics, counts)
    return reg[-1]


# ── OCR: assemble a fast-plate-ocr training set ────────────────────────────
def _harvest_ocr_rows(out_imgs: Path) -> int:
    """data/learning/plates/*.jpg + plates.jsonl -> (crop, text) rows."""
    pj = ROOT / "data" / "learning" / "plates.jsonl"
    if not pj.exists():
        return 0
    n = 0
    rows = []
    for ln in pj.read_text("utf-8", "ignore").splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        txt = "".join(ch for ch in str(r.get("text", "")).upper() if ch in ALPHABET)
        crop = ROOT / "data" / "learning" / "plates" / f"{r.get('id')}.jpg"
        if 4 <= len(txt) <= 12 and crop.exists():
            dst = out_imgs / f"harvest_{r['id']}.jpg"
            shutil.copy2(crop, dst)
            rows.append((dst.name, txt))
            n += 1
    (out_imgs.parent / "_harvest_rows.json").write_text(json.dumps(rows))
    return n


def assemble_ocr(src_root: Path, out: Path, seed: int = 0) -> dict:
    src = src_root / "anpr"
    if (out / "images").exists():
        shutil.rmtree(out / "images")
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []

    # 1. any CSV in the download with an image column + a text/plate column
    for cf in src.rglob("*.csv") if src.exists() else []:
        try:
            with open(cf, newline="", encoding="utf-8", errors="ignore") as f:
                rd = csv.DictReader(f)
                cols = {c.lower(): c for c in (rd.fieldnames or [])}
                icol = next((cols[k] for k in cols if "image" in k or "file" in k or "name" in k), None)
                tcol = next((cols[k] for k in cols if k in ("text", "plate", "label", "plate_number", "number")), None)
                if not (icol and tcol):
                    continue
                base = cf.parent
                for row in rd:
                    txt = "".join(ch for ch in str(row[tcol]).upper() if ch in ALPHABET)
                    if not (4 <= len(txt) <= 12):
                        continue
                    ip = next((p for p in base.rglob(Path(row[icol]).name)), None)
                    if ip is None:
                        continue
                    dst = out / "images" / f"csv_{len(rows)}_{ip.stem}.jpg"
                    shutil.copy2(ip, dst)
                    rows.append((dst.name, txt))
        except Exception:
            continue

    # 2. operator-confirmed crops from the running pipeline
    n_harvest = _harvest_ocr_rows(out / "images")
    hr = out / "_harvest_rows.json"
    if hr.exists():
        rows += [tuple(x) for x in json.loads(hr.read_text())]
        hr.unlink()

    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(20, int(len(rows) * 0.1)) if rows else 0
    with open(out / "annotations.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "plate_text", "split"])
        for i, (img, txt) in enumerate(rows):
            wr.writerow([img, txt, "val" if i < n_val else "train"])
    (out / "config.yaml").write_text(yaml.safe_dump(
        {"alphabet": ALPHABET + "_", "max_plate_slots": 12, "pad_char": "_",
         "img_height": 64, "img_width": 128}, sort_keys=False), encoding="utf-8")
    return {"n_total": len(rows), "n_val": n_val, "n_harvest": n_harvest}


def train_ocr(out: Path, ts: str) -> dict:
    """Fine-tune fast-plate-ocr if [train] is installed, else prepare + hand off."""
    try:
        import fast_plate_ocr.train                      # noqa: F401
    except Exception:
        return {"trained": False,
                "note": "fast-plate-ocr[train] not installed. Dataset is ready at "
                        f"{out}. Fine-tune on Kaggle (notebooks/train_anpr_kaggle.ipynb) "
                        "or: pip install \"fast-plate-ocr[train]\" and follow its "
                        "examples/fine_tune_workflow.ipynb, then copy the exported "
                        "model to models/plate_ocr/."}
    # fast-plate-ocr exposes a CLI; call it with the assembled dataset.
    dst = ROOT / "models" / "plate_ocr"
    dst.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "fast_plate_ocr.cli.train",
                        "--annotations", str(out / "annotations.csv"),
                        "--config", str(out / "config.yaml"),
                        "--output-dir", str(dst),
                        "--from-pretrained", "cct-s-v2-global-model"], check=True)
        return {"trained": True, "out": str(dst)}
    except Exception as e:
        return {"trained": False, "note": f"fast-plate-ocr train CLI failed ({e}); "
                f"dataset ready at {out} — fine-tune on Kaggle."}


# ── shared ────────────────────────────────────────────────────────────────
def _report(name: str, title: str, metrics, counts) -> None:
    rep = ROOT / "docs" / "learning_reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{name}.md").write_text(
        f"# ANPR — {title} — {name}\n\n"
        f"Counts: {counts}\n\nMetrics: {metrics}\n\n"
        f"Detector weights replace `models/license_plate_detector.pt`; OCR lands in "
        f"`models/plate_ocr/`. `models/anpr_registry.json` records every run.\n"
        f"Restart the server — `config.yaml anpr.ocr_backend/ocr_model` selects the OCR.\n"
        f"See docs/ANPR.md.\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune ANPR — plate detector + OCR")
    ap.add_argument("--detector", action="store_true", help="fine-tune the YOLO plate detector")
    ap.add_argument("--ocr", action="store_true", help="assemble/fine-tune the plate OCR")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--src", default=None, help="dataset root (default: weapon.dataset_root)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-per-source", type=int, default=8000)
    ap.add_argument("--min-images", type=int, default=300)
    args = ap.parse_args()
    if not (args.detector or args.ocr):
        ap.error("pass --detector and/or --ocr")

    src_root = Path(args.src) if args.src else _dataset_root()
    ts = time.strftime("%Y%m%d_%H%M%S")
    rc = 0

    if args.detector:
        out = src_root / "anpr" / "plate_yolo"
        print(f"[detector] assembling {src_root / 'anpr'} → {out}")
        counts = assemble_detector(src_root, out, args.max_per_source)
        print(f"  train={counts['n_train']}  val={counts['n_val']}  per-source={counts['per_source']}")
        if counts["n_train"] < args.min_images:
            print(f"  only {counts['n_train']} (< --min-images {args.min_images}) — "
                  f"run: python training/datasets_fetch.py --anpr")
            _emit({"state": "failed", "mode": "detector", **counts}); return 1
        data_yaml = write_data_yaml(out)
        if args.dry_run:
            print(f"  --dry-run: dataset ready at {data_yaml}")
            _emit({"state": "done", "mode": "detector", "dry_run": True, **counts})
        else:
            ft = train_detector(data_yaml, {"epochs": args.epochs, "batch": args.batch,
                                            "imgsz": args.imgsz}, ts)
            metrics = evaluate_detector(BASE_PT, ft, data_yaml)
            print(f"  mAP50-95  stock {metrics['base']['mAP50_95']} → ft {metrics['ft']['mAP50_95']} "
                  f"(Δ {metrics['delta']:+.4f})")
            entry = register_detector(ft, metrics, counts, ts)
            print(f"  registered → models/license_plate_detector.pt  ({entry['name']})")
            _emit({"state": "done", "mode": "detector", "metrics": metrics, **counts})

    if args.ocr:
        out = src_root / "anpr" / "ocr_set"
        print(f"[ocr] assembling crops+labels → {out}")
        oc = assemble_ocr(src_root, out)
        print(f"  {oc['n_total']} plate crops ({oc['n_harvest']} from data/learning/plates), "
              f"{oc['n_val']} held out")
        if args.dry_run or oc["n_total"] < 50:
            if oc["n_total"] < 50:
                print("  <50 labelled crops — ship the pretrained cct-s-v2-global-model "
                      "(config anpr.ocr_model) and collect more via the pipeline / Kaggle.")
            _emit({"state": "done", "mode": "ocr", "dry_run": args.dry_run, **oc})
        else:
            r = train_ocr(out, ts)
            print(f"  {r}")
            _report(f"anpr_ocr_{ts}", "Plate OCR (fast-plate-ocr)", r, oc)
            _emit({"state": "done", "mode": "ocr", **oc, **r})

    return rc


if __name__ == "__main__":
    sys.exit(main())
