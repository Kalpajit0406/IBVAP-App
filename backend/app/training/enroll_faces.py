"""
enroll_faces.py — build the face-recognition watchlist gallery
(models/face/gallery.json) from a folder of enrollment photos.

This is NOT a training run — it computes one reference embedding per identity
from a handful of clear photos (an "enrollment", the same step an attendance
kiosk does when it first learns a face) using the same insightface `buffalo_l`
pack `ibvap/face.py` FaceEngine loads at runtime, so the embeddings are
directly comparable to what the live pipeline produces.

    python training/enroll_faces.py --dry-run     # scan + print counts, write nothing
    python training/enroll_faces.py               # full enroll from the bundled photos
    python training/enroll_faces.py --src D:/other/photos \\
        --out models/face/gallery.json --thumbs-dir models/face/thumbs

Layout expected under --src: one subfolder per identity, named the way you
want it displayed. The default --src is the copy bundled inside the app
(package.ps1 carries it into every release), so the gallery can be rebuilt on
any machine the product is installed on:

    models/face/gallery_photos/
        adrika/*.jpg
        alice/*.jpg
        promita/*.jpg

To add someone to the watchlist, add a folder of their photos there and re-run
this script, then restart the server.

Each photo must show exactly one clear face; a photo with zero or more than
one detected face is skipped with a warning (bad enrollment input) rather than
silently guessing which face to use. Per-identity embeddings are the
L2-renormalized mean of each accepted photo's L2-normalized 512-d ArcFace
embedding — the standard "gallery centroid" practice for this embedding
family. The first accepted photo of each identity is copied to
--thumbs-dir/<id>.jpg so the console's read-only gallery listing
(GET /api/face/gallery) never needs `dataset_root` mounted at runtime.

See config.yaml `face:` and docs/FACE_RECOGNITION.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip()).strip("_") or "id"


def _load_face_app(model_root: Path, pack_name: str, device: str):
    from insightface.app import FaceAnalysis
    gpu = str(device) not in ("cpu", "-1")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                if gpu else ["CPUExecutionProvider"])
    app = FaceAnalysis(name=pack_name, root=str(model_root),
                       allowed_modules=["detection", "recognition"],
                       providers=providers)
    ctx_id = -1 if not gpu else (int(device) if str(device).isdigit() else 0)
    app.prepare(ctx_id=ctx_id, det_size=(320, 320))
    return app


def _embed_photo(app, path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"  SKIP  {path.name} — could not read image")
        return None
    faces = app.get(img)
    if len(faces) == 0:
        print(f"  SKIP  {path.name} — no face detected")
        return None
    if len(faces) > 1:
        print(f"  SKIP  {path.name} — {len(faces)} faces detected, need exactly 1 "
             "(crop to just this person, or remove other people from the photo)")
        return None
    emb = np.asarray(faces[0].normed_embedding, dtype=np.float32)
    return emb


def enroll(src: Path, model_root: Path, pack_name: str, device: str,
          threshold: float, dry_run: bool) -> dict:
    if not src.is_dir():
        raise SystemExit(f"--src not found: {src}")
    identity_dirs = sorted(p for p in src.iterdir() if p.is_dir())
    if not identity_dirs:
        raise SystemExit(f"no identity subfolders under {src} — expected one folder per person")

    app = None if dry_run else _load_face_app(model_root, pack_name, device)
    identities = []
    for d in identity_dirs:
        photos = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        print(f"{d.name}: {len(photos)} photo(s)")
        if dry_run:
            identities.append({"id": _slug(d.name), "name": d.name, "n_photos": len(photos),
                              "photos": [p.name for p in photos]})
            continue
        embs, first_photo = [], None
        for p in photos:
            emb = _embed_photo(app, p)
            if emb is not None:
                embs.append(emb)
                if first_photo is None:
                    first_photo = p
        if not embs:
            print(f"  ! {d.name}: zero usable photos — skipping this identity entirely")
            continue
        mean = np.mean(embs, axis=0)
        norm = float(np.linalg.norm(mean))
        mean = mean / norm if norm > 1e-6 else mean
        identities.append({
            "id": _slug(d.name), "name": d.name, "n_photos": len(embs),
            "embedding": mean.astype(np.float32).tolist(),
            "source_dir": str(d).replace("\\", "/"), "first_photo": str(first_photo),
        })
        print(f"  -> {len(embs)}/{len(photos)} photos used, embedding norm {np.linalg.norm(mean):.3f}")

    return {"identities": identities, "threshold": threshold}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=ROOT / "models/face/gallery_photos",
                    help="one subfolder per identity, full of clear face photos "
                         "(default: the enrollment photos bundled with the app)")
    ap.add_argument("--out", type=Path, default=ROOT / "models/face/gallery.json")
    ap.add_argument("--thumbs-dir", type=Path, default=ROOT / "models/face/thumbs")
    ap.add_argument("--model-root", type=Path, default=ROOT / "models/face",
                    help="contains models/<pack-name>/*.onnx — see ibvap/face.py")
    ap.add_argument("--pack-name", default="ibvap_face")
    ap.add_argument("--threshold", type=float, default=0.38,
                    help="cosine-similarity match threshold written into gallery.json "
                        "(face.match_threshold in config.yaml overrides this at runtime)")
    ap.add_argument("--device", default="0", help="CUDA device index, or 'cpu'")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and print photo counts per identity; write nothing")
    a = ap.parse_args()

    result = enroll(a.src, a.model_root, a.pack_name, a.device, a.threshold, a.dry_run)
    identities = result["identities"]

    if a.dry_run:
        total = sum(i["n_photos"] for i in identities)
        print(f"\ndry run: {len(identities)} identities, {total} photos total — nothing written")
        return

    if not identities:
        raise SystemExit("no identity had a usable photo — nothing to write")

    a.thumbs_dir.mkdir(parents=True, exist_ok=True)
    for ident in identities:
        fp = ident.pop("first_photo", None)
        if fp:
            try:
                shutil.copy2(fp, a.thumbs_dir / f"{ident['id']}.jpg")
            except OSError as e:
                print(f"  ! could not copy thumbnail for {ident['id']}: {e}")

    gallery = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_pack": a.pack_name, "embedding_dim": 512, "metric": "cosine",
        "match_threshold": a.threshold,
        "identities": identities,   # each: id, name, n_photos, embedding, source_dir
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(gallery, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}  ({len(identities)} identities: "
         f"{', '.join(i['name'] for i in identities)})")
    print(f"thumbnails in {a.thumbs_dir}")
    print("\nRestart the server (or hot-swap the detector) to pick up the new gallery.")


if __name__ == "__main__":
    main()
