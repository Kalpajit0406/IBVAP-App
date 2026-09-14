"""
datasets_fetch.py — pull the datasets IBVAP trains on into E:/Projects.

Downloaded datasets are large and machine-specific, so they live OUTSIDE the
repo, under `weapon.dataset_root` in config.yaml (default
`E:/Projects/ibvap-datasets/`). Only the trained `.pt` + a small report land
back in the repo.

    python training/datasets_fetch.py --weapon                 # YouTube-GDD (gun detection)
    python training/datasets_fetch.py --weapon --dry-run       # print the plan, download nothing
    python training/datasets_fetch.py --weapon --from D:\some\already-downloaded\YouTube-GDD

    python training/datasets_fetch.py --thermal               # LLVIP night/IR surveillance frames
    python training/datasets_fetch.py --thermal --roboflow-key KEY   # + FLIR ADAS thermal (person+vehicles)
    python training/datasets_fetch.py --thermal --kaggle             # + FLIR ADAS v2 via kaggle.json

    python training/datasets_fetch.py --anpr                  # Indian licence-plate sets (detector + OCR)
    python training/datasets_fetch.py --anpr --roboflow-key KEY  # + a Roboflow plate set
    python training/datasets_fetch.py --anpr --dry-run

YouTube-GDD (github.com/UCAS-GYX/YouTube-GDD): 5000 images, YOLO labels, classes
person(0) / gun(1). Images are one Google Drive zip; labels ship in the repo.

LLVIP (github.com/bupt-ai-cz/LLVIP): 15,488 infrared frames from fixed night
surveillance cameras, VOC-XML person labels, one Google Drive zip. Non-commercial
research licence. FLIR ADAS (Teledyne) adds thermal vehicles but needs a free
Roboflow key or a Kaggle token — the script falls back to LLVIP-only without one.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]

YT_GDD_REPO = "https://github.com/UCAS-GYX/YouTube-GDD"
YT_GDD_GDRIVE_ID = "1TH6kSx7WoFRrUPbxcDGYBrFrYUI1ReWa"   # from the repo README
YT_GDD_GDRIVE_URL = f"https://drive.google.com/file/d/{YT_GDD_GDRIVE_ID}/view"

# LLVIP processed dataset (infrared + visible + Annotations/*.xml), from
# github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md
LLVIP_GDRIVE_ID = "1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv"
LLVIP_GDRIVE_URL = f"https://drive.google.com/file/d/{LLVIP_GDRIVE_ID}/view"
# FLIR ADAS on Roboflow Universe (already YOLOv8-format). workspace/project/version:
FLIR_RF = ("thermal-imaging-0hwfw", "flir-data-set", 14)
FLIR_KAGGLE = "samdazel/teledyne-flir-adas-thermal-dataset-v2"

# ANPR — Indian licence-plate detection (YOLO) + plate OCR (crop + text).
ANPR_KAGGLE = [
    "deepakat002/indian-vehicle-number-plate-yolo-annotation",
    "barkataliarbab/license-plate-detection-dataset-10125-images",
    "dataclusterlabs/indian-number-plates-dataset",
]
ANPR_RF = ("roboflow-universe-projects", "license-plates-us-eu", 1)   # generic top-up


def _dataset_root() -> Path:
    try:
        cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
        r = (cfg.get("weapon", {}) or {}).get("dataset_root")
        if r:
            return Path(r)
    except Exception:
        pass
    return Path("E:/Projects/ibvap-datasets")


def _scan_counts(path: Path) -> dict:
    """Count images / label files anywhere under `path` (layout-agnostic)."""
    imgs = [p for p in path.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            and "images" in {q.name.lower() for q in p.parents}]
    lbls = [p for p in path.rglob("*.txt")
            if "labels" in {q.name.lower() for q in p.parents}]
    return {"images": len(imgs), "labels": len(lbls)}


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def fetch_youtube_gdd(root: Path, gdrive_id: str, assume_yes: bool,
                      dry_run: bool, src: str | None) -> int:
    dest = root / "youtube-gdd"
    print(f"YouTube-GDD  →  {dest}")
    print(f"  source : {YT_GDD_REPO}  +  Google Drive zip")
    print(f"  gdrive : {YT_GDD_GDRIVE_URL}")
    print(f"  size   : ~1-2 GB once extracted (exact size unknown until fetch)")

    if src:
        sp = Path(src)
        if not sp.exists():
            print(f"  --from path does not exist: {sp}")
            return 1
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  copying an existing copy from {sp} …")
        if not dry_run:
            for sub in ("images", "labels", "configs"):
                if (sp / sub).exists():
                    shutil.copytree(sp / sub, dest / sub, dirs_exist_ok=True)
        print("  counts:", _scan_counts(dest) if dest.exists() else "(dry-run)")
        return 0

    if dry_run:
        print("  --dry-run: nothing downloaded.")
        print("  next:     python training/datasets_fetch.py --weapon")
        return 0

    if not _confirm(f"Download YouTube-GDD into {dest}? [y/N] ", assume_yes):
        print("  aborted.")
        return 1

    dest.mkdir(parents=True, exist_ok=True)

    # 1. the repo — it ships the YOLO label zips (labels_only_gun.zip etc.) and
    #    the fold config. The Google Drive zip is IMAGES ONLY. Small; non-fatal
    #    if git is absent.
    repo_dir = dest / "_repo"
    if not repo_dir.exists():
        try:
            subprocess.run(["git", "clone", "--depth", "1", YT_GDD_REPO, str(repo_dir)],
                           check=True)
        except Exception as e:
            print(f"  (git clone skipped: {e})")

    # 2. the IMAGE zip from Google Drive via gdown
    try:
        import gdown
    except ImportError:
        print("\n  `gdown` is not installed — cannot fetch the Google Drive zip.")
        print("  Run:")
        print("     pip install gdown")
        print(f"     gdown {gdrive_id} -O \"{dest / 'youtube-gdd.zip'}\"")
        print(f"     (then unzip it into {dest})")
        return 2

    zip_path = dest / "youtube-gdd.zip"
    if not zip_path.exists():
        print(f"  downloading image zip via gdown (id={gdrive_id}) …")
        gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)
    if not any(dest.rglob("images/train/*")):
        print("  extracting images …")
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest)
        except zipfile.BadZipFile:
            print(f"  {zip_path} is not a valid zip — the Drive link may need a "
                  f"manual 'confirm large download'. Download it by hand into {dest}.")
            return 2

    # 3. the dataset root holding images/ (the zip nests a YouTube-GDD/ dir)
    img_dirs = [p.parent.parent for p in dest.rglob("images/train")][:1]
    ds = img_dirs[0] if img_dirs else dest

    # 4. YOLO labels — labels_only_gun.zip → ds/labels/{train,val}, plus the
    #    separate test-split labels. Gun is class 0 in the gun-only set.
    if repo_dir.exists() and not (ds / "labels").exists():
        _extract_labels(repo_dir, ds)

    # 5. the fold config, in case a consumer wants it
    if (repo_dir / "configs").exists():
        shutil.copytree(repo_dir / "configs", ds / "configs", dirs_exist_ok=True)

    counts = _scan_counts(ds)
    print(f"  done — {counts['images']} images, {counts['labels']} label files under {ds}")
    if counts["labels"] == 0:
        print("  NOTE: no labels — expected labels_only_gun.zip in the cloned repo. "
              f"Check {repo_dir}.")
    print(f"\n  next: python training/train_weapon.py --dry-run")
    return 0


def _extract_labels(repo_dir: Path, ds: Path) -> None:
    """labels_only_gun.zip (train/val) + YouTube-GDD_test_labels.zip (test) →
    ds/labels/{train,val,test}/*.txt. Drops a .gun_class marker (gun = 0)."""
    out = ds / "labels"
    out.mkdir(parents=True, exist_ok=True)
    tmp = ds / "_labels_tmp"

    def _pull(zname: str) -> int:
        zp = repo_dir / zname
        if not zp.exists():
            return 0
        if tmp.exists():
            shutil.rmtree(tmp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(tmp)
        n = 0
        for txt in tmp.rglob("*.txt"):
            rel = txt.parts
            split = next((s for s in ("train", "val", "test") if s in rel), None)
            if split is None:
                continue
            (out / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(txt, out / split / txt.name)
            n += 1
        shutil.rmtree(tmp, ignore_errors=True)
        return n

    n = _pull("labels_only_gun.zip")
    n += _pull("YouTube-GDD_test_labels.zip")
    (out / ".gun_class").write_text("0", encoding="utf-8")
    print(f"  labels: {n} .txt files → {out}  (gun = class 0)")


def _llvip_voc_note(ds: Path) -> None:
    """LLVIP ships VOC XML — the actual XML->YOLO conversion is train_thermal.py's
    job (it also folds in FLIR). Here we just confirm the pieces are present."""
    inf = next((p for p in ds.rglob("infrared") if p.is_dir()), None)
    ann = next((p for p in ds.rglob("Annotations") if p.is_dir()), None)
    n_img = len(list(inf.rglob("*.jpg"))) if inf else 0
    n_xml = len(list(ann.rglob("*.xml"))) if ann else 0
    print(f"  LLVIP: {n_img} infrared frames, {n_xml} VOC annotations under {ds}")
    if not (inf and ann):
        print("  NOTE: expected infrared/ + Annotations/ — check the extracted layout.")


def fetch_thermal(root: Path, roboflow_key: str | None, use_kaggle: bool,
                  assume_yes: bool, dry_run: bool, src: str | None) -> int:
    llvip = root / "llvip"
    flir = root / "flir"
    print(f"Thermal / night datasets  →  {root}")
    print(f"  LLVIP  : {LLVIP_GDRIVE_URL}   (~4 GB, person, fixed night IR cams)")
    print(f"  FLIR   : Roboflow {'/'.join(map(str, FLIR_RF))}  or  Kaggle {FLIR_KAGGLE}")
    print(f"           (thermal vehicles; needs a free key — LLVIP-only without one)")

    if src:
        sp = Path(src)
        if not sp.exists():
            print(f"  --from path does not exist: {sp}")
            return 1
        llvip.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            shutil.copytree(sp, llvip, dirs_exist_ok=True)
        _llvip_voc_note(llvip)
        return 0

    if dry_run:
        print("  --dry-run: nothing downloaded.")
        print("  next:     python training/train_thermal.py --dry-run")
        return 0

    if not _confirm(f"Download LLVIP (~4 GB) into {llvip}? [y/N] ", assume_yes):
        print("  aborted.")
        return 1

    # 1. LLVIP via gdown
    llvip.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        print("\n  `gdown` missing. Run:  pip install gdown")
        print(f"     gdown {LLVIP_GDRIVE_ID} -O \"{llvip / 'LLVIP.zip'}\"  (then unzip into {llvip})")
        return 2
    zip_path = llvip / "LLVIP.zip"
    if not zip_path.exists():
        print("  downloading LLVIP via gdown …")
        gdown.download(id=LLVIP_GDRIVE_ID, output=str(zip_path), quiet=False)
    if not any(llvip.rglob("infrared")):
        print("  extracting LLVIP …")
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(llvip)
        except zipfile.BadZipFile:
            print(f"  {zip_path} is not a valid zip — download it by hand into {llvip}.")
            return 2
    _llvip_voc_note(llvip)

    # 2. FLIR ADAS (optional — needs a key)
    if roboflow_key:
        try:
            from roboflow import Roboflow
            ws, proj, ver = FLIR_RF
            flir.mkdir(parents=True, exist_ok=True)
            print(f"  downloading FLIR ADAS from Roboflow ({ws}/{proj} v{ver}) …")
            (Roboflow(api_key=roboflow_key).workspace(ws).project(proj)
                .version(ver).download("yolov8", location=str(flir)))
            print(f"  FLIR: {_scan_counts(flir)} under {flir}")
        except Exception as e:
            print(f"  FLIR/Roboflow failed ({e}) — continuing LLVIP-only.")
    elif use_kaggle:
        kj = Path.home() / ".kaggle" / "kaggle.json"
        if not kj.exists():
            print(f"  --kaggle: {kj} not found. Put your Kaggle API token there.")
        else:
            try:
                flir.mkdir(parents=True, exist_ok=True)
                subprocess.run(["kaggle", "datasets", "download", "-d", FLIR_KAGGLE,
                                "-p", str(flir), "--unzip"], check=True)
                print(f"  FLIR (Kaggle): {_scan_counts(flir)} under {flir}  "
                      f"(COCO JSON — train_thermal.py converts it)")
            except Exception as e:
                print(f"  FLIR/Kaggle failed ({e}) — continuing LLVIP-only.")
    else:
        print("\n  FLIR skipped (no --roboflow-key / --kaggle). To add thermal "
              "vehicles later:")
        print(f"    python training/datasets_fetch.py --thermal --roboflow-key <your free roboflow key>")

    print(f"\n  next: python training/train_thermal.py --dry-run")
    return 0


def fetch_anpr(root: Path, roboflow_key: str | None, use_kaggle: bool,
               assume_yes: bool, dry_run: bool, src: str | None) -> int:
    dest = root / "anpr"
    print(f"ANPR datasets  →  {dest}")
    print(f"  Kaggle : {', '.join(ANPR_KAGGLE)}")
    print(f"  Roboflow (optional, --roboflow-key): {'/'.join(map(str, ANPR_RF))}")
    print(f"  also folded in at train time: data/learning/plates/ (operator-confirmed crops)")

    if src:
        sp = Path(src)
        if not sp.exists():
            print(f"  --from path does not exist: {sp}"); return 1
        (dest / "kaggle").mkdir(parents=True, exist_ok=True)
        if not dry_run:
            shutil.copytree(sp, dest / "kaggle", dirs_exist_ok=True)
        print("  counts:", _scan_counts(dest))
        return 0

    if dry_run:
        print("  --dry-run: nothing downloaded.")
        print("  add the Kaggle sets above via `kaggle datasets download` (needs "
              "~/.kaggle/kaggle.json), or run this in a Kaggle notebook.")
        print("  next: python training/train_anpr.py --detector --dry-run")
        return 0

    if not _confirm(f"Download ANPR datasets into {dest}? [y/N] ", assume_yes):
        print("  aborted."); return 1

    kdir = dest / "kaggle"
    kdir.mkdir(parents=True, exist_ok=True)
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if kj.exists() or use_kaggle:
        for slug in ANPR_KAGGLE:
            name = slug.split("/")[-1]
            if (kdir / name).exists():
                continue
            try:
                subprocess.run(["kaggle", "datasets", "download", "-d", slug,
                                "-p", str(kdir / name), "--unzip"], check=True)
            except Exception as e:
                print(f"  {slug} failed ({e})")
        print(f"  Kaggle: {_scan_counts(kdir)} under {kdir}")
    else:
        print(f"\n  no ~/.kaggle/kaggle.json — download the Kaggle sets by hand:")
        for slug in ANPR_KAGGLE:
            print(f"    kaggle datasets download -d {slug} -p \"{kdir / slug.split('/')[-1]}\" --unzip")

    if roboflow_key:
        try:
            from roboflow import Roboflow
            ws, proj, ver = ANPR_RF
            (Roboflow(api_key=roboflow_key).workspace(ws).project(proj)
                .version(ver).download("yolov8", location=str(dest / "roboflow")))
            print(f"  Roboflow: {_scan_counts(dest / 'roboflow')}")
        except Exception as e:
            print(f"  Roboflow top-up failed ({e}) — continuing.")

    print(f"\n  next: python training/train_anpr.py --detector --dry-run")
    return 0


def fetch_vehicle_type(root: Path, assume_yes: bool, dry_run: bool,
                       src: str | None) -> int:
    """Best-effort only: unlike --weapon/--thermal/--anpr, there is no known,
    verified public dataset covering all 8 vehicle-type classes — auto_rickshaw
    and tanker in particular have no common analogue on Kaggle/Roboflow that
    could be named here with any confidence. This deliberately does NOT
    hardcode dataset slugs it can't vouch for (unlike ANPR_KAGGLE above).

    The reliable path is training/build_vehicle_type_dataset.py, which crops
    real vehicle footage this deployment has already harvested
    (data/learning/) into a coarse starting pool for hand-sorting. --from lets
    you register an already-downloaded external set (e.g. something you found
    and vetted yourself on Kaggle/Roboflow) alongside it instead.
    """
    dest = root / "vehicle_type_external"
    print(f"Vehicle-type dataset  →  {dest}")
    print("  No bundled dataset list here — auto-rickshaw/tanker coverage on "
          "public sets is unverified. Recommended:")
    print("    python training/build_vehicle_type_dataset.py "
          "  # bootstraps from this deployment's own footage")
    print("  If you've already found/vetted an external dataset yourself, "
          "register it with --from <path>.")

    if src:
        sp = Path(src)
        if not sp.exists():
            print(f"  --from path does not exist: {sp}"); return 1
        if dry_run:
            print(f"  --dry-run: would copy {sp} -> {dest}")
            return 0
        if not _confirm(f"Copy {sp} into {dest}? [y/N] ", assume_yes):
            print("  aborted."); return 1
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sp, dest, dirs_exist_ok=True)
        print(f"  registered under {dest} — still needs hand-sorting into "
              f"training/train_vehicle_type.py's train/<class>/ + val/<class>/ layout.")
        return 0

    if dry_run:
        print("  --dry-run: nothing downloaded.")
        return 0

    print("  nothing to fetch automatically — run build_vehicle_type_dataset.py "
          "or pass --from <path>.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch IBVAP training datasets into E:/Projects")
    ap.add_argument("--weapon", action="store_true",
                    help="fetch YouTube-GDD (gun / held-gun detection)")
    ap.add_argument("--thermal", action="store_true",
                    help="fetch LLVIP (+ optional FLIR ADAS) for a night/thermal fine-tune")
    ap.add_argument("--anpr", action="store_true",
                    help="fetch Indian licence-plate datasets (detector + OCR)")
    ap.add_argument("--vehicle-type", action="store_true",
                    help="best-effort: no verified dataset for all 8 classes exists, "
                         "see training/build_vehicle_type_dataset.py instead")
    ap.add_argument("--dest", default=None,
                    help="dataset root (default: weapon.dataset_root in config.yaml)")
    ap.add_argument("--from", dest="src", default=None,
                    help="register an already-downloaded dataset dir instead of downloading")
    ap.add_argument("--gdrive-id", default=YT_GDD_GDRIVE_ID,
                    help="override the Google Drive file id for the YouTube-GDD image zip")
    ap.add_argument("--roboflow-key", default=None,
                    help="thermal: free Roboflow API key, to also pull FLIR ADAS (YOLO format)")
    ap.add_argument("--kaggle", action="store_true",
                    help="also pull the optional key-gated set (FLIR for --thermal) via ~/.kaggle/kaggle.json")
    ap.add_argument("--yes", action="store_true", help="skip the download confirmation")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    args = ap.parse_args()

    root = Path(args.dest) if args.dest else _dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    print(f"dataset root: {root}\n")

    if args.thermal:
        return fetch_thermal(root, args.roboflow_key, args.kaggle,
                             args.yes, args.dry_run, args.src)
    if args.anpr:
        return fetch_anpr(root, args.roboflow_key, args.kaggle,
                          args.yes, args.dry_run, args.src)
    if args.vehicle_type:
        return fetch_vehicle_type(root, args.yes, args.dry_run, args.src)
    if args.weapon:
        return fetch_youtube_gdd(root, args.gdrive_id, args.yes, args.dry_run, args.src)
    ap.error("nothing to do — pass --weapon, --thermal, --anpr or --vehicle-type")


if __name__ == "__main__":
    sys.exit(main())
