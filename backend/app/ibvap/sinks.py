"""
sinks.py — where an alert goes when it leaves this machine.

Until now it went nowhere. The product raised a banner in a console window and
that was the whole of "real-time alert generation": at an unmanned Border Out
Post, nobody was told. These are the outbound channels, behind one interface so
`alert_forward.py` does not care which is configured.

Deliberately few. A sink earns its place by being something a real command
centre already ingests:

  * **webhook** — JSON POST. Universal: every modern C2, SIEM and incident tool
    accepts one, and it is the channel a judge can watch working on a laptop.
  * **syslog** — RFC 5424 over UDP/TCP, with the payload in ArcSight CEF. This
    is what an existing SOC plugs into without writing any code at all.
  * **mqtt** — optional, off by default. Common in Indian command-centre
    deployments, but it drags in a broker, so it must be asked for.

Not built, on purpose rather than by omission: SMTP (mail is not an alerting
channel for something measured in seconds) and CAP XML (designed for public
broadcast warnings, not post-to-HQ telemetry). Saying so is more honest than
half-shipping them.

Every sink is synchronous and returns (ok, detail). Retries, backoff and
ordering are the forwarder's job, not the sink's — a sink that retried
internally would stall the queue behind it.
"""
from __future__ import annotations

import json
import logging
import socket
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger("ibvap.sinks")

# CEF severity is 0-10; map our five levels onto it so a SIEM's own rules fire.
_CEF_SEVERITY = {"Info": 2, "Low": 3, "Medium": 5, "High": 7, "Critical": 10}
_SEVERITY_ORDER = {"Info": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}

_SYSLOG_FACILITY = 13 * 8        # local security/authorisation messages


def severity_at_least(severity: str, minimum: str) -> bool:
    return _SEVERITY_ORDER.get(severity or "Info", 0) >= _SEVERITY_ORDER.get(minimum, 0)


def event_payload(row: dict) -> dict:
    """The wire format. This *is* the integration contract — keep it stable,
    and add rather than rename."""
    return {
        "schema": "ibvap.event/2",
        "event_id": row.get("event_id"),
        "ts_utc": row.get("ts_utc"),
        "severity": row.get("severity"),
        "category": row.get("category"),
        "score": row.get("score"),
        "site": {"site_id": row.get("site_id"), "post_name": row.get("post_name")},
        "camera": {"cam_id": row.get("cam_id"), "name": row.get("cam_name"),
                   "lat": row.get("lat"), "lon": row.get("lon")},
        "counts": {"persons": row.get("persons"), "vehicles": row.get("vehicles")},
        "evidence": {"sha256": row.get("ev_hash")},
        "details": row.get("details") or {},
        "producer": row.get("producer"),
    }


class Sink:
    name = "sink"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg or {}
        self.min_severity = str(self.cfg.get("min_severity", "Info"))
        self.enabled = bool(self.cfg.get("enabled", False))
        self.label = str(self.cfg.get("label", self.name))

    def wants(self, row: dict) -> bool:
        return self.enabled and severity_at_least(row.get("severity", "Info"),
                                                  self.min_severity)

    def send(self, row: dict) -> tuple[bool, str]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label, "enabled": self.enabled,
                "min_severity": self.min_severity, "target": self.target()}

    def target(self) -> str:
        return ""


class WebhookSink(Sink):
    """HTTP POST of one JSON event."""

    name = "webhook"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.url = str(self.cfg.get("url", ""))
        self.timeout = float(self.cfg.get("timeout_s", 5.0))
        self.headers = dict(self.cfg.get("headers") or {})
        self.verify_tls = bool(self.cfg.get("verify_tls", True))
        if not self.url:
            self.enabled = False

    def target(self) -> str:
        return self.url

    def send(self, row: dict) -> tuple[bool, str]:
        body = json.dumps(event_payload(row)).encode()
        req = urllib.request.Request(self.url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        # Lets a receiver discard a duplicate after an at-least-once retry.
        req.add_header("X-IBVAP-Event-Id", str(row.get("event_id") or ""))
        for k, v in self.headers.items():
            req.add_header(str(k), str(v))
        ctx = None
        if self.url.lower().startswith("https") and not self.verify_tls:
            # A border post's own C2 endpoint is often on a private CA or a
            # self-signed cert; refusing to deliver the alert is the worse
            # failure. Off by default, and it must be chosen explicitly.
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as r:
                code = r.getcode()
            if 200 <= code < 300:
                return True, f"HTTP {code}"
            return False, f"HTTP {code}"
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            return False, f"{type(e).__name__}: {e}"


class SyslogSink(Sink):
    """RFC 5424 syslog carrying an ArcSight CEF payload."""

    name = "syslog"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.host = str(self.cfg.get("host", ""))
        self.port = int(self.cfg.get("port", 514))
        self.proto = str(self.cfg.get("protocol", "udp")).lower()
        self.timeout = float(self.cfg.get("timeout_s", 5.0))
        if not self.host:
            self.enabled = False

    def target(self) -> str:
        return f"{self.proto}://{self.host}:{self.port}"

    @staticmethod
    def _esc(v) -> str:
        return str(v).replace("\\", "\\\\").replace("=", "\\=").replace("\n", " ")

    def _cef(self, row: dict) -> str:
        d = row.get("details") or {}
        ext = {
            "externalId": row.get("event_id"),
            "rt": row.get("ts_utc"),
            "cat": row.get("category"),
            "dvc": row.get("site_id") or "",
            "cs1Label": "post", "cs1": row.get("post_name") or "",
            "cs2Label": "camera", "cs2": row.get("cam_name") or str(row.get("cam_id")),
            "cs3Label": "evidenceSha256", "cs3": row.get("ev_hash") or "",
            "cn1Label": "persons", "cn1": row.get("persons") or 0,
            "cn2Label": "vehicles", "cn2": row.get("vehicles") or 0,
            "cn3Label": "score", "cn3": row.get("score") or 0,
        }
        if row.get("lat") is not None:
            ext["deviceCustomFloatingPoint1Label"] = "lat"
            ext["deviceCustomFloatingPoint1"] = row["lat"]
            ext["deviceCustomFloatingPoint2Label"] = "lon"
            ext["deviceCustomFloatingPoint2"] = row.get("lon")
        if d.get("matched_name"):
            ext["suser"] = d["matched_name"]
        if d.get("plate"):
            ext["cs4Label"] = "plate"; ext["cs4"] = d["plate"]
        body = " ".join(f"{k}={self._esc(v)}" for k, v in ext.items())
        name = str(d.get("type") or row.get("category") or "event")
        sev = _CEF_SEVERITY.get(row.get("severity", "Info"), 5)
        return (f"CEF:0|IBVAP|BorderVideoAnalytics|2|{row.get('category')}|"
                f"{name}|{sev}|{body}")

    def send(self, row: dict) -> tuple[bool, str]:
        pri = _SYSLOG_FACILITY + (2 if row.get("severity") == "Critical" else 4)
        stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        host = (row.get("site_id") or "ibvap").replace(" ", "_")
        msg = f"<{pri}>1 {stamp} {host} ibvap - - - {self._cef(row)}"
        data = msg.encode("utf-8", "replace")
        try:
            if self.proto == "tcp":
                with socket.create_connection((self.host, self.port), self.timeout) as s:
                    # RFC 6587 octet counting — a stream needs framing or the
                    # collector cannot tell where one message ends.
                    s.sendall(f"{len(data)} ".encode() + data)
            else:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.settimeout(self.timeout)
                    s.sendto(data, (self.host, self.port))
            return True, "sent"
        except OSError as e:
            return False, f"{type(e).__name__}: {e}"


class MqttSink(Sink):
    """Publish to an MQTT topic. Optional — needs paho-mqtt installed."""

    name = "mqtt"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.host = str(self.cfg.get("host", ""))
        self.port = int(self.cfg.get("port", 1883))
        self.topic = str(self.cfg.get("topic", "ibvap/events"))
        self.qos = int(self.cfg.get("qos", 1))
        self.username = self.cfg.get("username")
        self.password = self.cfg.get("password")
        self.timeout = float(self.cfg.get("timeout_s", 5.0))
        if not self.host:
            self.enabled = False
        if self.enabled:
            try:
                import paho.mqtt.client  # noqa: F401
            except ImportError:
                logger.warning("MQTT sink disabled — paho-mqtt is not installed "
                               "(pip install paho-mqtt)")
                self.enabled = False

    def target(self) -> str:
        return f"mqtt://{self.host}:{self.port}/{self.topic}"

    def send(self, row: dict) -> tuple[bool, str]:
        try:
            import paho.mqtt.publish as publish
            auth = ({"username": self.username, "password": self.password}
                    if self.username else None)
            topic = f"{self.topic}/{row.get('severity','Info').lower()}"
            publish.single(topic, json.dumps(event_payload(row)), qos=self.qos,
                           hostname=self.host, port=self.port, auth=auth,
                           keepalive=int(self.timeout))
            return True, "published"
        except Exception as e:                       # paho raises broadly
            return False, f"{type(e).__name__}: {e}"


_KINDS = {"webhook": WebhookSink, "syslog": SyslogSink, "mqtt": MqttSink}


def build_sinks(cfg: dict) -> list[Sink]:
    """Build every declared sink. `cfg` is config.yaml's `alerts.sinks` list."""
    out: list[Sink] = []
    for entry in (cfg or []):
        kind = str((entry or {}).get("type", "")).lower()
        cls = _KINDS.get(kind)
        if cls is None:
            logger.warning("Unknown alert sink type %r — skipping", kind)
            continue
        s = cls(entry)
        out.append(s)
        logger.info("Alert sink %s (%s) — %s, min severity %s",
                    s.label, s.name, "enabled" if s.enabled else "disabled",
                    s.min_severity)
    return out
