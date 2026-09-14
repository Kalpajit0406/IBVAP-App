# IBVAP docs

Start with **[`../CLAUDE.md`](../CLAUDE.md)** — the full architecture, measured
numbers, and design rationale. Everything below is a focused guide for one part.

## Setup & operations
| doc | about |
|---|---|
| [STARTUP.md](STARTUP.md) | Bring a new machine up from zero — CUDA/PyTorch, deps, first run |
| [CCTV_INTEGRATION.md](CCTV_INTEGRATION.md) | Wire real RTSP / NVR / ONVIF cameras into the pipeline; vendor URL tables |
| [CUSTOM_DOMAIN.md](CUSTOM_DOMAIN.md) | Serve the phone link on your own domain via a cloudflared named tunnel |

## Features
| doc | about |
|---|---|
| [GEOFENCE.md](GEOFENCE.md) | Virtual fences — operator-drawn polygon zones / tripwire lines, image-space breach → Critical |
| [INTRUSION.md](INTRUSION.md) | Per-camera evidence snapshots on breach / weapon / crouch-lying, SHA-256 bound into the chain; Grid ⇄ Focus operator layout |
| [ANPR.md](ANPR.md) | Number-plate recognition — plate detector on vehicle crops + threaded EasyOCR |
| [WEAPON_DETECTION.md](WEAPON_DETECTION.md) | Trained `gun` model (`training/train_weapon.py`), held-gun confirmation, fused with AIM posture |
| [THERMAL_NIGHT.md](THERMAL_NIGHT.md) | Night/IR yolo26n fine-tune (`training/train_thermal.py`, LLVIP + FLIR ADAS) → hot-swap model |
| [CONTINUOUS_LEARNING.md](CONTINUOUS_LEARNING.md) | Harvest → operator review → `training/retrain.py` fine-tune + hot-swap; per-camera calibration |
| [SCREEN_WATCH.md](SCREEN_WATCH.md) | `scripts/screen_watch.py` — the full stack painted as a transparent on-screen overlay (demo) |

## Reference / submission material
| doc | about |
|---|---|
| [IBVAP_Project_Context_TechStack_1.md](IBVAP_Project_Context_TechStack_1.md) | Full project context, architecture decisions, hardware cost math |
| [IBVAP_Technical_Specifications_1.md](IBVAP_Technical_Specifications_1.md) | Submission-ready numbers with `[HARD]` / `[TUNABLE]` / `[MISSING]` tags |
| [IBVAP_References.md](IBVAP_References.md) | Numbered citations (R1–R22) for every hard number |
| [AUDIT.md](AUDIT.md) | Code-audit findings and fixes |

`learning_reports/` holds the per-run mAP / precision-recall reports written by
the training scripts (`weapon_<ts>.md`, `thermal_<ts>.md`, `posture_<ts>.md`).
