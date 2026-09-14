# IBVAP — Setup & Operations Guide

IBVAP (border-surveillance video analytics) ships as **one standalone Windows
product**: the Command Client (`ibvap_app.exe`) plus its own Python backend and
a complete Python + CUDA runtime, all in one folder. Nothing else needs to be
installed on the machine that runs it — no separate Python, no pip installs,
no CUDA toolkit.

This guide covers the packaged release. If you're working from source instead,
see `backend/README.md` (how the bundle is built) and `README.md` (Flutter
project layout).

---

## 1. What's in the release folder

```
Release/
├── ibvap_app.exe            the Command Client — run this
├── flutter_windows.dll, data\   Flutter runtime + assets (do not touch)
└── backend/
    ├── app/                  the server: server.py, ibvap/, models/, config.yaml,
    │                         static/, test_videos/, data/ (its own evidence store)
    └── runtime/              a self-contained Python 3.11 + CUDA/torch/ultralytics
                              environment — python.exe here is NOT your system Python
```

Copy the **whole `Release` folder** to move or back up the product — a USB
drive, another machine, a network share. It is self-contained; nothing inside
references a path outside itself.

**Requirements on the target machine:**
- Windows 10/11, 64-bit.
- An NVIDIA GPU (CUDA) is strongly recommended — the bundled runtime is built
  for CUDA 12.6 and the pipeline is designed to always run on GPU. It will
  fall back to CPU if no GPU is found, but expect a large drop in frame rate.
- No internet connection is required to run it. (Internet is only used if you
  turn on remote access — §7 — or re-train a model.)

---

## 2. First launch

1. Double-click `ibvap_app.exe`.
2. The console opens with the backend **off**. Click **Turn On Backend** for
   the default (**All Sources**), or the small dropdown beside it to pick a
   different startup mode:

   | Mode | What it does |
   |---|---|
   | **All Sources (config.yaml)** | Both real cameras (from `config.yaml`) and phone-camera slots. Normal operating mode. |
   | **Phone Cameras (WebSockets)** | Only opens the phone-intake listener (`:8443`) — use this for a live demo with phones and nothing else. |
   | **Real CCTV / RTSP** | Only pulls the RTSP/webcam/file streams configured in `config.yaml`, ignoring phone slots. |
   | **Screen Capture (CAM-00)** | Grabs this PC's own screen as a synthetic camera — for a quick demo with no camera at all. |
   | **4-Camera Demo Replay** | Replays the four bundled sample videos (`test_videos/`) as 4 synchronized cameras — for a demo with no camera or phone at hand. |

3. Once the backend reports **ONLINE** in the header, the **Monitor** tab
   shows live video as soon as a camera or phone connects.

The very first backend start on a given machine takes noticeably longer (auto
warm-up passes for the detector, pose and plate models); after that it's fast.

---

## 3. The eight tabs

Navigate with the left rail, or `Ctrl+1`…`Ctrl+8`.

| # | Tab | What it's for |
|---|---|---|
| 1 | **Monitor** | The live mosaic. Grid / Focus layout, virtual-fence drawing, the alert log, continuous-learning panel. Right-click a camera tile (or use the **Streams** button) for per-camera actions: reconnect, or **remove stream**. |
| 2 | **Overview** | Model hot-swap, pipeline throughput stats, per-camera live/idle/offline state. |
| 3 | **Cameras** | Add/edit RTSP, webcam, file, or phone-slot streams. |
| 4 | **Models** | Switch detection model (Nano/Medium/Thermal/fine-tuned), review continuous-learning candidates, trigger retraining. |
| 5 | **Snapshots** | Intrusion evidence: fence breaches, confirmed weapons, malicious posture — filterable, zoomable, annotated ⇄ raw toggle. |
| 6 | **ANPR** | Vehicle arrivals: plate text, vehicle type, confidence, thumbnail. |
| 7 | **Evidence** | The hash-chained event log (`GET /api/evidence/verify`) — tamper-evident record of every Critical/High event, breach, weapon and plate. |
| 8 | **Settings** | Server address, API token, remote-access (tunnel) setup, alarm sound, keyboard shortcuts. |

**Keyboard shortcuts:** `Ctrl+1`–`8` switch tabs, `F5` forces an immediate
refresh, `Ctrl+K` acknowledges the oldest pending alarm, `Ctrl+Shift+K`
acknowledges all of them.

---

## 4. Adding cameras

**Real CCTV / IP camera / webcam / video file** — go to **Cameras**, click
**Add CCTV / RTSP Camera**, and fill in:

- **URL** — `rtsp://user:pass@ip:554/stream1`, a plain webcam index (`0`, `1`,
  …), a video file path (for a demo/test loop), or `screen` for a synthetic
  screen-capture camera.
- **Zone sensitivity** (0–1) — how readily this camera's zone drives the risk
  score up when someone is in it.
- **Transport** (`tcp`/`udp`) and **decode FPS** for RTSP sources.

The stream loads immediately — no restart needed.

**Phone camera** — leave a slot's URL as `ws` (the default for an unconfigured
slot), then on the phone open `https://<this-PC's-LAN-IP>:8443/cam/<id>` in a
browser and allow camera access. On the **Cameras** tab, **Connect Phone (QR
Code)** shows the exact address and a QR code to scan. The phone's browser
will show a certificate warning on first connect — this is expected (the
server uses a self-signed certificate for the HTTPS phone listener) and is
safe to accept.

**Removing a stream** — right-click its tile in Monitor (or use the
**Streams** button), then **Remove stream…**. This closes the capture, hangs
up a connected phone, and removes it from `config.yaml`. Recorded evidence and
snapshots from that camera are kept. Adding the stream again from **Cameras**
restores it.

---

## 5. Alert levels — what they mean

| Level | Meaning |
|---|---|
| **Normal** | Nothing notable. |
| **High** | People/vehicles present (however many, at night, or in a sensitive zone), an unusual posture (crouching, lying, arms up), or an unconfirmed two-handed "aim" stance. Worth a look, not an alarm. |
| **Critical** | A **confirmed weapon** (object-detected gun, held by a tracked person and persisted across several frames), an **armed threat** (that plus an aim posture on the same person), or a **virtual-fence breach**. These raise the console alarm and require acknowledgement (`Ctrl+K`). |

A detected person alone — however many, at night, anywhere — never raises
Critical by itself. This is a deliberate calibration: it keeps the alarm
meaningful instead of firing on ordinary foot traffic.

---

## 6. Virtual fences

Draw a **Zone** (polygon) or **Line** (tripwire) on a camera's own view in
Monitor. A zone breach fires when a person/vehicle is inside it; a line fires
when one crosses it, with an optional direction (`a→b`, `b→a`, or both). Every
breach is hash-chained into the evidence log with a snapshot of the moment.

Demo fences ship pre-configured on CAM-00 and CAM-01 (`data/fences.json`) —
delete or disable them from the Monitor sidebar if you don't want a person
walking through the frame to trigger a Critical breach.

---

## 7. Remote access (Cloudflare Tunnel)

By default nothing is reachable from outside the machine — this is a
standalone local product, and that's the right posture for a fielded system.
For a demo where a phone or a remote viewer needs to reach it from off the
LAN, **Settings → Remote access · Cloudflare Tunnel** publishes it over a
trusted HTTPS address with no port forwarding or firewall rule.

1. Turn the backend on first (the tunnel needs something to point at).
2. Choose **Quick** (free, no account — gets a random `*.trycloudflare.com`
   address that changes every time you start it) or **Named** (a tunnel you
   created in your own Cloudflare account, on your own domain — paste its
   hostname and token).
3. Choose what to expose:
   - **Phone cameras only** — only the phone-intake page and its video
     WebSocket are reachable; the dashboard, API and all controls stay hidden.
     Safe to leave running during a demo.
   - **Full console** — the entire dashboard and API become reachable at that
     address. The app warns and asks for confirmation before publishing this,
     especially since `security.require_token` is `false` by default (see
     §8) — anyone with the link could then change cameras, fences, or shut the
     backend down.
4. Click **Start tunnel**. Once it shows **TUNNEL UP**, scan the QR code or
   share the printed link.

cloudflared must be installed once per machine:
```
winget install --id Cloudflare.cloudflared
```
(or point Settings at an existing `cloudflared.exe`). The tunnel's own log is
saved to `%APPDATA%\ibvap_app\tunnel.log` if something needs diagnosing.

---

## 8. Security posture (read before a real deployment)

The product ships **pre-configured for local testing with relaxed security**
(`backend/app/config.yaml`):

```yaml
network:
  bind_host: "0.0.0.0"        # reachable from the LAN, not just this PC
  lan_phone_intake: true      # phones can stream in

security:
  require_token: false        # any client can issue write commands
  allow_loopback_writes: true
```

**Before a real deployment, or once phone testing is done:**

```yaml
network:
  bind_host: "127.0.0.1"      # this machine only
  lan_phone_intake: false

security:
  require_token: true
```

With `require_token: true`, every write request (adding a camera, editing a
fence, switching models, shutting the backend down) must carry the token
found in `backend/app/data/api_token` in an `X-IBVAP-Token` header — the app
does this automatically once the token is entered in **Settings**. Read-only
requests (`/status`, viewing streams) are never gated.

If you also disabled the firewall rule for the phone-testing phase, remove it
once done:
```powershell
Remove-NetFirewallRule -DisplayName "IBVAP phone test"
```

---

## 9. Evidence & integrity

Every Critical/High level change, fence breach, confirmed weapon, and
identified plate is appended to a SHA-256 hash chain
(`backend/app/data/hash_chain.jsonl`) — each record embeds the previous
record's hash, so any tampering breaks the chain from that point forward.
Verify it at any time:

```
cd backend\app
..\runtime\Scripts\python.exe main.py --verify-chain
```
or just open the **Evidence** tab, which verifies automatically and shows
either "records verified · no tampering detected" or exactly which record the
chain breaks at (`GET /api/evidence/verify` is the same check over the API).

---

## 10. ANPR (number-plate recognition)

Vehicle plates are read automatically — no configuration needed — whenever a
vehicle enters a camera's view: the plate is read across several frames as
the vehicle approaches, and a result is only shown once enough independent
reads agree (avoiding a single blurry frame deciding a wrong plate). Results
appear in the **ANPR** tab with the plate text, vehicle type, and confidence;
the underlying data is `data/anpr_results/<cam>/` (one photo + one JSON record
per vehicle) and `data/plates/plates.csv` (a flat log).

This has been calibrated and measured against real street footage — not just
synthetic test images (see `backend/app/docs/ANPR.md` for the method and the
measured accuracy). It reads Indian-format plates (`SS RR L(L)(L) NNNN`, and
the BH series) by construction; `anpr.region: "XX"` in `config.yaml` switches
to a generic alphanumeric format for other regions.

---

## 11. Model switching & continuous learning

**Models** tab → pick a detection profile (**Nano** is the default, fast
profile; **Medium** trades speed for accuracy; a **Thermal/night** model is
included for IR cameras). Switching is hot — no restart.

The pipeline harvests borderline detections while running; review them under
**Models → Continuous learning**, mark Keep/Drop, and trigger a retrain
(`Build set (dry-run)` first to preview, then `Retrain`). A newly fine-tuned
model appears in the model list with its measured mAP delta over the base.

---

## 12. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Backend won't turn on | Another process already has ports 8090/8443 open — check for a leftover `python.exe` from a previous crash and stop it, or check `%TEMP%` for the backend's own log if `BackendManager` reported a failure. |
| Phone can't connect | Confirm the phone is on the same Wi-Fi/LAN as this PC (unless using the Cloudflare tunnel — §7), and that `network.lan_phone_intake: true` in `config.yaml`. A Windows Firewall prompt on first phone-testing session must be allowed, or add the rule manually (see §8). |
| Camera shows "SIGNAL LOST" | The RTSP/webcam source stopped sending frames; the app auto-reconnects. Use **Cameras → Reconnect** to force it immediately. |
| Cloudflare tunnel times out | Usually a network blocking the QUIC protocol tunnel uses first — the app automatically retries over HTTP/2 on port 443, which gets through almost any firewall. If it still fails, check `%APPDATA%\ibvap_app\tunnel.log`. |
| Alarm won't stop sounding | An alert is unacknowledged — `Ctrl+K` (oldest) or `Ctrl+Shift+K` (all), or click **ACK** on the banner. |
| Need to reset everything | Delete `%APPDATA%\ibvap_app\config.json` (app settings only — no camera/evidence data lost) and relaunch. |

---

## 13. Updating this deployment

To move to a newer build: copy the new `Release` folder over, **but preserve
`backend\app\data\`** (the evidence database, hash chain, snapshots, fences,
and plate history) from the old copy first — this folder is the product's
operational record and isn't something a rebuild should discard.
