# Screen-share detection overlay (`scripts/screen_watch.py`) — demo only

Point IBVAP at your laptop screen. It grabs a region of the desktop and runs the
**same analysis stack the camera pipeline uses** — **YOLO26n** detector +
**yolo26n-pose** skeletons + trained **firearm detector**
(`models/weapon_detector.pt`, `ibvap/weapon.py`) + **ANPR** number-plate
recognition (`ibvap/anpr.py`) — then paints the results straight back onto the
screen as a **transparent, click-through overlay** locked over that region. Play
a clip or open a photo underneath and IBVAP "sees" what is on screen:

- **green box + 17-point skeleton** on people
- **amber box** on vehicles (car, truck, bus, motorcycle, bicycle, …), with a
  **cyan plate readout** under the box once ANPR reads one (`--no-anpr` skips it;
  it also logs to `data/plates/plates.csv`, same as the server)
- **grey box** on other objects (bag, bottle, laptop, knife, phone, …)
- **red box `GUN xx%`** on a gun held by a tracked person, once it has persisted
  a few frames; a full-width **`GUN DETECTED`** / **`ARMED THREAT`** (gun + AIM
  posture on one person) banner. `--no-weapon` skips this.
- everything turns **red** when the posture rules flag `LYING` / `CROUCH` /
  `ARMS-UP` / `SCAN` / `AIM` (weapon-ready)
- a small HUD: fps, grab / infer ms, live person / vehicle / object counts

Also reachable from the launch menu: `python server.py` → **[2] Screen →
on-screen overlay**, or `python server.py --mode screen --surface overlay`.

It runs in its own process — no server, no muxer, no `/status`, and it does
**not** write the SHA-256 evidence chain. It reads `config.yaml` for the model /
pose / weapon / anpr settings, and (like the server) ANPR appends readings to
`data/plates/plates.csv` and `--learn` writes to `data/learning/`.

## Run

```bash
pip install mss          # one-time; tkinter already ships with Python on Windows

python scripts/screen_watch.py                       # whole primary monitor
python scripts/screen_watch.py --pick                # drag the watch rectangle first
python scripts/screen_watch.py --region 200,120,1280,720
python scripts/screen_watch.py --all                 # every COCO class, not the curated set
python scripts/screen_watch.py --no-pose --fps 20    # boxes only, faster
python scripts/screen_watch.py --debug --seconds 30  # print detections, auto-stop
```

**Esc** on the overlay, or **Ctrl+C** in the terminal, stops it.

| flag | |
|---|---|
| `--pick` | translucent full-screen picker — drag the area to watch |
| `--region X,Y,W,H` | fixed rectangle in screen pixels |
| `--monitor N` | `mss` monitor index (1 = primary) |
| `--fps N` | target detection rate (default 15; overlay always redraws ~25 fps) |
| `--infer-width N` | downscale width for the network (default 1100 — smaller = faster) |
| `--all` / `--classes 0,2,7` | widen or pin the detected classes (default: a curated person + vehicle + carry-item set) |
| `--no-pose` | skip the skeleton / posture pass |
| `--weights PATH` | default `yolo26n.pt` (a single frame, so the fixed-batch `.engine` is not worth it) |
| `--device 0` / `cpu` | |
| `--seconds N` | auto-stop (handy for a timed demo) |
| `--debug` | fps + detections to the terminal |

## How it works

```
mss.grab(region)  ─►  resize to --infer-width  ─►  blank our own HUD footprint
       │
       ▼   background thread, ~15-30 fps on an RTX 3050
YOLO26n .predict(classes=…)  ─►  supervision.ByteTrack  ─►  stable track ids
       │
       ├─ person boxes ─►  ibvap.posture.PoseEstimator.run_batch  ─►  17-kpt skeleton
       │                    ibvap.posture.PoseClassifier          ─►  LYING/CROUCH/AIM…
       ▼
scale boxes + keypoints back to overlay pixels  ─►  publish Snapshot
       │
       ▼   Tk main thread, ~25 fps
transparent borderless top-most window over the region  ─►  canvas.delete + redraw
```

- **Two threads**: a capture+detect worker and the Tk overlay. Tk is only ever
  touched by the main thread; the worker hands over one `Snapshot` under a lock.
- **Warm-up** primes the exact input shape in `_build()` so the first live frame
  isn't a ~2 s cuDNN-autotune stall.
- **Transparency** is Tk `-transparentcolor` (a near-black key colour): keyed
  pixels are not drawn and are click-through, so you keep controlling the video
  underneath. The drawn boxes / skeleton lines are thin and don't feed back into
  detection; the HUD chip's own footprint is blanked out of each frame before
  inference so it can't be detected as a "laptop".

## Demo tips

- Run the media in a **windowed** player, not exclusive full-screen — an
  exclusive-full-screen surface can render above a top-most window.
- `--pick` the exact video area so the HUD and window border sit on the desktop,
  not over the footage.
- A static photo gives a rock-steady overlay; on video the boxes step at the
  detection rate — raise `--fps` (and lower `--infer-width`) for smoother.
- Windows only for the overlay (`-transparentcolor`). The detector itself runs
  anywhere; on macOS/Linux the window would just be opaque.
