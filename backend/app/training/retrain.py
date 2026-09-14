"""
retrain.py — the deliberate step of the continuous-learning loop.

Assembles the operator-reviewed harvest pool into a YOLO dataset, fine-tunes
`yolo26n.pt` on it (backbone frozen + a slice of COCO-val mixed in against
catastrophic forgetting), measures base-vs-fine-tuned mAP on a held-out split,
and registers the new weights in `models/registry.json` so the dashboard can
hot-swap them.

    python training/retrain.py                 # detector fine-tune from the reviewed pool
    python training/retrain.py --dry-run       # assemble + print counts, no training
    python training/retrain.py --posture       # sweep AIM/posture thresholds vs. operator verdicts
    python training/retrain.py --anpr          # fine-tune the plate detector / build an OCR sub map

Nothing is applied automatically: the fine-tuned model appears as a new entry in
the model switcher and is loaded / rolled back by the operator. `--posture`
writes a report with *suggested* config values; it never edits config.yaml.
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
POSE_PT = "yolo26n-pose.pt"

# COCO 80 class names — the fine-tuned head keeps this shape so the hot-swap is
# drop-in (harvested labels only ever use ids 0,2,3,5,7).
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
EVAL_CLASSES = [0, 2, 3, 5, 7]


def _cfg() -> dict:
    return yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


def _emit(obj: dict) -> None:
    """The server tails stdout for this line."""
    print("RETRAIN_RESULT " + json.dumps(obj), flush=True)


# ── dataset assembly ───────────────────────────────────────────────────────
def _latest_verdicts(pool: Path) -> dict[str, dict]:
    v: dict[str, dict] = {}
    f = pool / "verdicts.jsonl"
    if f.exists():
        for ln in f.read_text("utf-8").splitlines():
            try:
                r = json.loads(ln)
                v[r["id"]] = r
            except (ValueError, KeyError):
                continue
    return v


def _manifest(pool: Path) -> dict[str, dict]:
    m: dict[str, dict] = {}
    f = pool / "manifest.jsonl"
    if f.exists():
        for ln in f.read_text("utf-8").splitlines():
            try:
                r = json.loads(ln)
                m[r["id"]] = r
            except (ValueError, KeyError):
                continue
    return m


def assemble_dataset(pool: Path, out: Path, val_fraction: float,
                     replay_images: int, seed: int = 0) -> dict:
    man = _manifest(pool)
    ver = _latest_verdicts(pool)
    items = []
    for _id, row in man.items():
        verdict = ver.get(_id, {}).get("verdict", "pending")
        if verdict not in ("keep", "background"):
            continue
        items.append((_id, row, verdict, row.get("ts", 0.0)))
    items.sort(key=lambda t: t[3])                      # chronological — split by time, no leakage
    n_val = max(1, int(len(items) * val_fraction)) if items else 0
    val_ids = {t[0] for t in items[-n_val:]} if n_val else set()

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    n = {"train": 0, "val": 0, "bg": 0}
    for _id, row, verdict, _ts in items:
        split = "val" if _id in val_ids else "train"
        img = ROOT / row["frame"]
        if not img.exists():
            continue
        shutil.copy2(img, out / "images" / split / f"{_id}.jpg")
        lbl_dst = out / "labels" / split / f"{_id}.txt"
        if verdict == "background":
            lbl_dst.write_text("")
            n["bg"] += 1
        else:
            corr = ver.get(_id, {}).get("boxes")
            if corr:
                lbl_dst.write_text("".join(
                    f"{int(b['cls'])} {b['xywhn'][0]:.6f} {b['xywhn'][1]:.6f} "
                    f"{b['xywhn'][2]:.6f} {b['xywhn'][3]:.6f}\n" for b in corr))
            else:
                src = ROOT / row["label"]
                lbl_dst.write_text(src.read_text() if src.exists() else "")
        n[split] += 1

    # replay a slice of COCO-val against catastrophic forgetting
    n_replay = 0
    if replay_images > 0:
        n_replay = _mix_coco_replay(out, replay_images, seed)

    return {"n_train": n["train"], "n_val": n["val"], "n_bg": n["bg"],
            "n_replay": n_replay, "n_kept": len(items)}


def _mix_coco_replay(out: Path, count: int, seed: int) -> int:
    """Copy a random sample of COCO-val images (person/vehicle) into train/.
    Ultralytics auto-downloads coco.yaml + coco8 style assets; here we use the
    small coco128 set that ships with ultralytics if present, else skip."""
    try:
        from ultralytics.utils import SETTINGS
        ds_dir = Path(SETTINGS.get("datasets_dir", ""))
    except Exception:
        return 0
    for cand in (ds_dir / "coco128", ds_dir / "coco8"):
        imgs = sorted((cand / "images" / "train2017").glob("*.jpg")) if cand.exists() else []
        if imgs:
            random.Random(seed).shuffle(imgs)
            k = 0
            for p in imgs[:count]:
                lbl = cand / "labels" / "train2017" / (p.stem + ".txt")
                shutil.copy2(p, out / "images" / "train" / f"coco_{p.stem}.jpg")
                (out / "labels" / "train" / f"coco_{p.stem}.txt").write_text(
                    lbl.read_text() if lbl.exists() else "")
                k += 1
            return k
    return 0


def write_data_yaml(out: Path) -> Path:
    y = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
         "nc": 80, "names": COCO_NAMES}
    p = out / "data.yaml"
    p.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return p


# ── training / eval / register ─────────────────────────────────────────────
def train(data_yaml: Path, params: dict, ts: str) -> Path:
    from ultralytics import YOLO
    if not BASE_PT.exists():
        raise SystemExit(f"{BASE_PT} not found — the fine-tune base must be a .pt "
                         f"(a .engine/.onnx cannot be trained).")
    m = YOLO(str(BASE_PT))
    m.train(data=str(data_yaml), epochs=int(params["epochs"]),
            imgsz=int(params["imgsz"]), freeze=int(params["freeze"]),
            device=0, project=str(ROOT / "runs" / "detect"), name=f"ibvap_ft_{ts}",
            exist_ok=True, verbose=False,
            workers=0,          # Windows: dataloader-worker processes crash under a busy GPU
            batch=int(params.get("batch", 8)), cache=False, plots=False)
    save_dir = Path(getattr(m.trainer, "save_dir", ROOT / "runs" / "detect" / f"ibvap_ft_{ts}"))
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
                "mAP50_95": round(float(r.box.map), 4)}
    base, ft = _m(base_pt), _m(ft_pt)
    return {"base": base, "ft": ft,
            "delta": round(ft["mAP50_95"] - base["mAP50_95"], 4)}


def register(ft_pt: Path, metrics: dict, n_images: int, ts: str) -> dict:
    reg_path = ROOT / _cfg().get("learning", {}).get("paths", {}).get(
        "registry", "models/registry.json")
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    dst = ROOT / "models" / f"yolo26n_ft_{ts}.pt"
    shutil.copy2(ft_pt, dst)
    entry = {"name": f"ft_{ts}", "weights": f"models/{dst.name}",
             "pose_weights": POSE_PT, "created": time.time(),
             "label": f"FT {ts}", "base_mAP": metrics["base"],
             "ft_mAP": metrics["ft"], "delta": metrics["delta"], "n_images": n_images}
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
    return entry


# ── posture threshold sweep (--posture) ────────────────────────────────────
def sweep_posture(pool: Path) -> dict:
    ver = _latest_verdicts(pool)
    rows = []
    f = pool / "posture.jsonl"
    if f.exists():
        for ln in f.read_text("utf-8").splitlines():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
    labelled = [r for r in rows if ver.get(r.get("id"), {}).get("verdict") in ("keep", "drop")]
    report = ROOT / "docs" / "learning_reports"
    report.mkdir(parents=True, exist_ok=True)
    out = report / f"posture_{time.strftime('%Y%m%d_%H%M%S')}.md"
    if len(labelled) < 20:
        out.write_text(f"# Posture threshold sweep\n\nNot enough reviewed posture "
                       f"samples ({len(labelled)} < 20). Confirm/Discard more AIM "
                       f"alerts in the dashboard first.\n")
        return {"labelled": len(labelled), "report": str(out), "swept": False}
    # (sweep implementation is a follow-up; report the data we have)
    keep = sum(1 for r in labelled if ver[r["id"]]["verdict"] == "keep")
    out.write_text(
        f"# Posture threshold sweep\n\n{len(labelled)} reviewed samples "
        f"({keep} confirmed, {len(labelled)-keep} rejected).\n\n"
        f"Grid-search over `pose.aim_*` against these verdicts is the next step; "
        f"the labelled data is in `data/learning/posture.jsonl` + `verdicts.jsonl`.\n")
    return {"labelled": len(labelled), "report": str(out), "swept": False}


def fine_tune_plates(pool: Path) -> dict:
    ver = _latest_verdicts(pool)
    rows = []
    f = pool / "plates.jsonl"
    if f.exists():
        for ln in f.read_text("utf-8").splitlines():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
    kept = [r for r in rows if ver.get(r.get("id"), {}).get("verdict") == "keep"]
    return {"plate_samples": len(rows), "kept": len(kept), "trained": False,
            "note": "ANPR training moved to training/train_anpr.py "
                    "(--detector / --ocr). These crops are folded in there. "
                    "See docs/ANPR.md."}


# ── entrypoint ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="IBVAP continuous-learning fine-tune")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--posture", action="store_true")
    ap.add_argument("--anpr", action="store_true")
    args = ap.parse_args()

    cfg = _cfg()
    lc = cfg.get("learning", {}) or {}
    pool = ROOT / lc.get("paths", {}).get("pool", "data/learning")
    ds_out = ROOT / lc.get("paths", {}).get("dataset", "datasets/ibvap_live")
    rp = lc.get("retrain", {}) or {}
    ts = time.strftime("%Y%m%d_%H%M%S")

    if args.posture:
        r = sweep_posture(pool)
        print(f"posture: {r}")
        _emit({"state": "done", "mode": "posture", **r})
        return 0
    if args.anpr:
        r = fine_tune_plates(pool)
        print(f"anpr: {r}")
        _emit({"state": "done", "mode": "anpr", **r})
        return 0

    print(f"Assembling dataset from {pool} → {ds_out} …")
    counts = assemble_dataset(pool, ds_out, float(rp.get("val_fraction", 0.2)),
                              int(rp.get("replay_images", 300)))
    print(f"  kept={counts['n_kept']}  train={counts['n_train']}  val={counts['n_val']}  "
          f"background={counts['n_bg']}  coco_replay={counts['n_replay']}")

    min_images = int(rp.get("min_images", 40))
    if counts["n_kept"] < min_images:
        print(f"Only {counts['n_kept']} kept frames (< min_images {min_images}). "
              f"Keep more candidates in the dashboard.")
        _emit({"state": "failed", "mode": "detector", "reason": "too_few_images",
               "n_images": counts["n_kept"]})
        return 1

    data_yaml = write_data_yaml(ds_out)
    if args.dry_run:
        print(f"--dry-run: dataset ready at {data_yaml}. Not training.")
        _emit({"state": "done", "mode": "detector", "dry_run": True, **counts})
        return 0

    print(f"Fine-tuning {BASE_PT.name}  epochs={rp.get('epochs', 40)}  "
          f"freeze={rp.get('freeze', 10)} …")
    ft_pt = train(data_yaml, rp, ts)
    print(f"  best weights: {ft_pt}")
    print("Evaluating base vs fine-tuned on the held-out val split …")
    metrics = evaluate(BASE_PT, ft_pt, data_yaml)
    print(f"  base mAP50-95 {metrics['base']['mAP50_95']}  →  "
          f"ft {metrics['ft']['mAP50_95']}  (Δ {metrics['delta']:+.4f})")
    entry = register(ft_pt, metrics, counts["n_kept"], ts)
    print(f"Registered as '{entry['name']}' → {entry['weights']}")
    print("Switch to it from the dashboard MODEL row; switch to Nano to roll back.")
    _emit({"state": "done", "mode": "detector", "name": entry["name"],
           "base_mAP": metrics["base"], "ft_mAP": metrics["ft"],
           "delta": metrics["delta"], "n_images": entry["n_images"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
