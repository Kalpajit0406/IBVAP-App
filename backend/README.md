# backend/ — the bundled, standalone IBVAP Python runtime

This folder makes the IBVAP Command Client a **standalone product**: everything
the app needs to run the actual video-analytics pipeline — the server code, the
trained models, and a complete Python + CUDA/torch/ultralytics environment —
ships right next to `ibvap_app.exe`. Nothing needs to be installed separately
on the machine that runs it.

```
backend/
├── app/          vendored copy of the IBVAP server (E:\IBVAP) — server.py,
│                 ibvap/, scripts/, tools/, training/, static/, config.yaml,
│                 models/*.pt, root yolo26*.pt/.engine, test_videos/, cert.pem
│                 + key.pem. NOT a symlink — a real, independent copy, so this
│                 product keeps working even if the original E:\IBVAP dev
│                 checkout is moved, renamed, or deleted.
└── runtime/      a fully self-contained Python 3.11 install: python.exe,
                  its own python311.dll/vcruntime DLLs, its own copy of the
                  standard library (Lib/, DLLs/, tcl/ — not just a venv
                  pointing back at wherever Python happens to be installed
                  on this machine), and every third-party package the
                  backend needs (torch/CUDA 12.6, ultralytics, opencv,
                  fastapi, easyocr, fast-plate-ocr, tensorrt, …). Invoked
                  directly by absolute path with PYTHONHOME set to this
                  folder (see BackendManager.start() in
                  ../lib/services/backend_manager.dart) — never needs
                  "activating", and needs no Python installed anywhere else
                  on the machine.
```

### Why `runtime/` has its own `Lib/`, `DLLs/`, `tcl/` and not just `Scripts/`

A plain `python -m venv` on Windows is **not** self-contained even with
`--copies`: `Scripts\python.exe` is a thin launcher that, at process start,
hands off to the *base* interpreter named in `pyvenv.cfg`'s `home` — which
also supplies `python311.dll`, the standard library, and the `DLLs\*.pyd`
extension modules. A bare venv therefore keeps a hard dependency on whatever
Python install it was created from continuing to exist at that exact path.
To make this bundle genuinely independent of the machine's own Python:

1. `python311.dll`, `python3.dll`, `vcruntime140*.dll` are copied directly
   into `runtime\Scripts\` (next to `python.exe`) — Windows' DLL search order
   checks an executable's own directory first, so the launcher no longer
   needs to reach back to the base install for its runtime DLL.
2. The base install's `Lib\` (excluding `site-packages`, which the venv
   already has its own copy of), `DLLs\`, and `tcl\` are copied into
   `runtime\`.
3. `BackendManager` launches with the environment variable
   `PYTHONHOME=<...>\backend\runtime` set. This is what actually redirects
   `sys.path` / `sys.base_prefix` to the local copies instead of the system
   install — without it, step 2's files are just inert (Python's `sys.path`
   construction is hardcoded to `pyvenv.cfg`'s `home`, not to whatever the
   venv's own `Lib\` happens to contain).

One residual, both harmless and effectively unavoidable short of switching to
python.org's separate "embeddable" distribution: the Windows venv launcher
mechanism still creates a brief child-process handoff to the base install's
`python.exe` **binary** at startup (confirmed via `Get-CimInstance
Win32_Process`) — but with the above in place that child correctly reports
`sys.base_prefix` / loads every package from `runtime\`, not the system
install, so no package, stdlib file, or version from the system Python is
ever actually used. The dependency that remains is only on that one `.exe`
file continuing to exist — not on its packages, its stdlib, or on `E:\IBVAP`.

`ibvap_app`'s [`AppConfig`](../lib/config/app_config.dart) looks for this
folder next to its own executable and, if found, points `backendDir` /
`pythonPath` at it automatically (`AppConfig.instance.isBundled` is `true`).
The Settings screen shows a **BUNDLED** badge when this is active, and a
"Reset to bundled" button to get back to it after pointing at something else.

## How this was built

```powershell
# 1. A clean, purpose-built Python 3.11 venv — not the machine's general
#    Python install (which typically has a lot of unrelated packages).
python -m venv backend\runtime

# 2. Torch pinned to the exact version IBVAP was developed and tested
#    against — a different torch build is a real compatibility risk with a
#    pinned `ultralytics==8.4.137` and a hardware-specific TensorRT engine.
backend\runtime\Scripts\python.exe -m pip install torch==2.13.0 torchvision `
    --index-url https://download.pytorch.org/whl/cu126

# 3. Everything else IBVAP needs.
backend\runtime\Scripts\python.exe -m pip install -r backend\app\requirements.txt

# 4. TensorRT — only if you want to rebuild backend\app\yolo26n.engine for a
#    *different* GPU than the one it currently ships with (see below).
backend\runtime\Scripts\python.exe -m pip install "tensorrt-cu12==10.13.3.9" onnx onnxslim

# 5. Vendor the IBVAP source (from a separate checkout) into backend\app,
#    excluding dev/ephemeral state:
robocopy E:\IBVAP backend\app /E /XD .git .agent __pycache__ runs datasets data notebooks

# 6. Make the runtime independent of the machine's own Python install (see
#    "Why runtime/ has its own Lib/, DLLs/, tcl/" above) — $base is wherever
#    step 1's venv was created from.
$base = "C:\Users\<you>\AppData\Local\Programs\Python\Python311"   # your system Python
Copy-Item "$base\python311.dll","$base\python3.dll","$base\vcruntime140.dll","$base\vcruntime140_1.dll" backend\runtime\Scripts\
robocopy "$base\Lib" backend\runtime\Lib /E /XD site-packages __pycache__ /XF *.pyc
robocopy "$base\DLLs" backend\runtime\DLLs /E
robocopy "$base\tcl" backend\runtime\tcl /E
```

`BackendManager` sets `PYTHONHOME` to `backend\runtime` automatically once it
detects both `Lib\` and `DLLs\` present there (see `backend_manager.dart`) —
step 6 is what makes that kick in; skipping it still runs, just with a live
dependency on `$base` staying where it is.

## Re-running / rebuilding

- **Update the bundled app code** (after changing the IBVAP Python source):
  re-run the `robocopy` step above, then `..\package.ps1` to re-stage it into
  the built Release folder.
- **Rebuild the whole standalone package**: `..\package.ps1` from the Flutter
  project root — builds `flutter build windows --release` and mirrors this
  `backend\` folder into `build\windows\x64\runner\Release\backend\`.

## The TensorRT engine is machine-specific

`backend/app/yolo26n.engine` is a TensorRT build **pinned to the exact GPU and
driver it was exported on** — it is not portable to a different graphics card.
`config.yaml`'s `model.weights` points at it by default because that's the
fastest option *on the machine it was built for*. If you copy this standalone
package to a different machine and the engine fails to load, either:

- point `model.weights` in `backend\app\config.yaml` at `yolo26n.pt` instead
  (works anywhere with a CUDA GPU, or CPU — just slower), or
- re-export a new engine on the target machine:
  `backend\runtime\Scripts\python.exe backend\app\tools\export_engine.py`
  (needs step 4 above done first).

## Size

Expect roughly 6–9 GB total (`runtime/` dominates — CUDA torch alone is
~4 GB); `app/` (source + trained models + demo videos) is under 200 MB. Disk
size was an explicit non-concern for this build.
