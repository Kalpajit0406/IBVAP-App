# IBVAP Command Client (Windows)

A native Flutter desktop client for the **IBVAP** border-surveillance server
(`E:\IBVAP`). It talks to the server entirely over its HTTP API — no shared code.

## Features

| Screen | What it does |
|---|---|
| **Monitor** | Live MJPEG mosaic (`/stream`) with the virtual-fence overlay painted on top. Grid ⇄ Focus layout switch with 1–4 main cameras (`POST /api/layout`). Click the mosaic to place fence points. Right sidebar: **Alert log** (risk edges, breaches, plates, + snapshot thumbnails), **Virtual fences** (draw / edit / enable-disable / delete / clear-camera, per stream, zone or tripwire, targets + direction), **Continuous learning** (candidate strip with Keep/Drop/Bg, retrain + dry-run, live train state). |
| **Overview** | Model hot-swap row with mAP deltas (`/api/models`, `/api/switch-model`). Pipeline stats. Camera rows with live/idle/offline + RTSP reconnect. Subsystem flags (anpr / weapon / geofence / learning / snapshots). |
| **Snapshots** | Grid of intrusion snapshots (`/api/snapshots` + `/snap/<cam>/<file>`) with camera + reason filters and a zoomable viewer that toggles annotated ⇄ raw. |
| **Settings** | Server base URL (persisted), `/status` poll interval, about. |

The server address is editable from the app bar (default `http://127.0.0.1:8090`)
and remembered between launches in `%APPDATA%\ibvap_app\config.json`.

## Layout

```
lib/
├── main.dart                     entry — loads config, starts AppState polling
├── theme.dart                    IbvapColors + buildIbvapTheme()
├── config/app_config.dart        server URL + poll interval, JSON-file persisted (no plugins)
├── api/ibvap_client.dart         every server endpoint: status, devices, models, fences,
│                                 snapshots, learn pool/review/calibration/retrain, layout,
│                                 switch-model, reconnect, + stream / snapshot / thumb URLs
├── models/
│   ├── server_status.dart        typed view over /status
│   ├── fence.dart                Fence + MosaicLayout / MosaicTile (canvas↔camera mapping)
│   └── alert.dart                AlertEntry (kind, colour, icon)
├── state/
│   ├── app_state.dart            /status @1Hz + /api/models + /api/snapshots timers; owns:
│   ├── alert_log.dart            derives the alert log from status edges + snapshots
│   └── fence_editor.dart         fence list + draw/edit draft, shared by Monitor & sidebar
├── widgets/
│   ├── mjpeg_view.dart           multipart-JPEG (`multipart/x-mixed-replace`) stream widget
│   ├── fence_overlay.dart        CustomPainter for fences + draft; taps → fence points
│   ├── model_switcher.dart       the MODEL row
│   ├── connection_badge.dart · server_bar.dart · panel.dart
└── screens/
    ├── home_screen.dart          NavigationRail shell
    ├── monitor_screen.dart       the mosaic stage + layout bar + draw bar + sidebar
    └── panels/                   overview · snapshots · settings · alerts_section ·
                                  fences_section · learning_section
```

## Run

```
cd "E:\IBVAP app"
flutter pub get
flutter run -d windows           # or: flutter build windows --release
```

Point it at a running IBVAP server (`python server.py --mode all` in `E:\IBVAP`,
then a phone or `python scripts/feed_test.py --src test_videos --cams 3`).

`flutter analyze` is clean and `flutter test` passes (shell smoke test + mosaic
mapping + fence JSON round-trip).
