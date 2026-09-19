"""
alert_sink.py — a standalone receiver that proves alerts leave the machine.

Stands in for the command-and-control system IBVAP forwards to. Run it in a
second window, point `alerts.sinks[].url` at it, and every alert appears here
the moment it fires — which is the whole claim "real-time alert generation"
makes, and the one thing a console banner cannot demonstrate, because a banner
never left the process that drew it.

    python scripts/alert_sink.py                 # listen on :9000
    python scripts/alert_sink.py --port 9100
    python scripts/alert_sink.py --fail          # refuse everything, to show
                                                 # the queue build up and drain

Open http://127.0.0.1:9000/ in a browser for the live list.

This is a demo and test tool, not part of the product: no auth, no TLS, keeps
the last few hundred alerts in memory. Do not expose it beyond the laptop.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

RECEIVED: deque = deque(maxlen=500)
LOCK = threading.Lock()
FAIL = False

COLOUR = {"Critical": "\033[1;31m", "High": "\033[1;33m", "Medium": "\033[36m",
          "Low": "\033[32m", "Info": "\033[90m"}
RESET = "\033[0m"

PAGE = """<!doctype html><meta charset=utf-8>
<title>IBVAP alert sink</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#0f1216;color:#e6edf3}
 header{padding:14px 18px;background:#161b22;border-bottom:1px solid #30363d}
 h1{margin:0;font-size:16px}
 .sub{color:#8b949e;font-size:12px;margin-top:2px}
 table{border-collapse:collapse;width:100%}
 td,th{padding:8px 12px;border-bottom:1px solid #21262d;text-align:left;
       vertical-align:top;font-size:13px}
 th{color:#8b949e;font-weight:600;font-size:11px;text-transform:uppercase;
    letter-spacing:.04em}
 .sev{font-weight:700;white-space:nowrap}
 .Critical{color:#ff6b6b}.High{color:#ffa657}.Medium{color:#79c0ff}
 .Low{color:#7ee787}.Info{color:#8b949e}
 code{color:#8b949e;font-size:11px}
 .empty{padding:40px 18px;color:#8b949e}
</style>
<header>
  <h1>IBVAP alert sink</h1>
  <div class=sub>Standing in for a command-and-control endpoint ·
    <span id=n>0</span> alert(s) received · auto-refreshes</div>
</header>
<div id=out class=empty>Waiting for the first alert…</div>
<script>
async function tick(){
  const r = await fetch('/alerts'); const rows = await r.json();
  document.getElementById('n').textContent = rows.length;
  const out = document.getElementById('out');
  if(!rows.length) return;
  out.className='';
  out.innerHTML = '<table><tr><th>Received</th><th>Severity</th><th>Category</th>'
    + '<th>Site / camera</th><th>Detail</th><th>Event</th></tr>' + rows.map(a=>{
    const e=a.event||{};
    const d=e.details||{};
    const det=[d.matched_name&&('name: '+d.matched_name),d.plate&&('plate: '+d.plate),
               d.fence&&('fence: '+d.fence),d.posture&&('posture: '+d.posture),
               d.elapsed_s&&('dwell: '+d.elapsed_s+'s'),d.type]
              .filter(Boolean).join(' · ');
    const cam=(e.camera||{});const site=(e.site||{});
    return `<tr><td>${a.at}</td>`
      + `<td class="sev ${e.severity}">${e.severity||'?'}</td>`
      + `<td>${e.category||''}</td>`
      + `<td>${site.post_name||site.site_id||''}<br><code>${cam.name||('cam '+cam.cam_id)}</code></td>`
      + `<td>${det||''}</td>`
      + `<td><code>${(e.event_id||'').slice(-10)}</code></td></tr>`;
  }).join('') + '</table>';
}
tick(); setInterval(tick, 1000);
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/alerts"):
            with LOCK:
                body = json.dumps(list(RECEIVED)[::-1]).encode()
            self._send(200, body, "application/json")
        else:
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        if FAIL:
            # Simulates an unreachable C2 so the operator can watch the queue
            # grow and then drain in order once it recovers.
            self._send(503, b'{"error":"sink is in --fail mode"}', "application/json")
            print(f"{COLOUR['Critical']}REFUSED{RESET} (--fail) "
                  f"{self.headers.get('X-IBVAP-Event-Id','?')[-10:]}", flush=True)
            return
        try:
            ev = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            self._send(400, b'{"error":"not json"}', "application/json")
            return
        at = datetime.now().strftime("%H:%M:%S")
        with LOCK:
            RECEIVED.append({"at": at, "event": ev})
        sev = str(ev.get("severity", "?"))
        cam = (ev.get("camera") or {})
        site = (ev.get("site") or {})
        d = ev.get("details") or {}
        extra = " ".join(f"{k}={d[k]}" for k in
                         ("matched_name", "plate", "fence", "posture", "elapsed_s")
                         if d.get(k))
        print(f"{at}  {COLOUR.get(sev,'')}{sev:<9}{RESET}"
              f"{str(ev.get('category','')):<12}"
              f"{str(site.get('post_name') or site.get('site_id') or '-'):<16}"
              f"{str(cam.get('name') or cam.get('cam_id')):<14}{extra}",
              flush=True)
        self._send(200, b'{"ok":true}', "application/json")

    def log_message(self, *a) -> None:      # quiet; we print our own line
        pass


def main() -> None:
    global FAIL
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--fail", action="store_true",
                    help="reject every delivery, to demonstrate store-and-forward")
    a = ap.parse_args()
    FAIL = a.fail

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("=" * 72)
    print(f"  IBVAP alert sink listening on http://127.0.0.1:{a.port}")
    print(f"  Point config.yaml alerts.sinks[].url at "
          f"http://127.0.0.1:{a.port}/alert")
    if a.fail:
        print("  --fail: every delivery will be REFUSED (queue should build up)")
    print("=" * 72, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
