"""
security.py — write-access control for the HTTP API.

The server listens on 0.0.0.0 so phones and LAN browsers can reach it, which
used to mean anyone on the network could POST /api/shutdown, rewrite camera
URLs, delete fences or swap the model. Mutating API calls now need either:

  * a loopback source address (the desktop command client and the web
    dashboard running on the same machine), or
  * the API token in the ``X-IBVAP-Token`` header.

The token lives in ``security.token_file`` (default ``data/api_token``) and is
generated on first start, so a fresh install is locked down with no manual
step. Read-only GETs, the MJPEG streams and the phone camera intake stay open —
those are needed for pairing and viewing.

Loopback trust is only sound because server.py trusts proxy headers from
127.0.0.1 alone; with ``forwarded_allow_ips="*"`` any LAN client could claim to
be localhost via ``X-Forwarded-For``.
"""
from __future__ import annotations

import hmac
import ipaddress
import logging
import secrets
from pathlib import Path

logger = logging.getLogger("ibvap.security")

TOKEN_HEADER = "x-ibvap-token"
_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def load_or_create_token(path: str | Path) -> str:
    p = Path(path)
    try:
        if p.is_file():
            tok = p.read_text("utf-8").strip()
            if len(tok) >= 24:
                return tok
        p.parent.mkdir(parents=True, exist_ok=True)
        tok = secrets.token_urlsafe(32)
        p.write_text(tok + "\n", encoding="utf-8")
        logger.info("Generated API write token at %s", p)
        return tok
    except OSError as e:
        # No writable data dir: fall back to an in-memory token. Remote writes
        # are then impossible until the disk is fixed, which is the safe side.
        logger.error("Cannot persist API token (%s) — remote writes disabled", e)
        return secrets.token_urlsafe(32)


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return host == "localhost"


def needs_token(method: str, path: str) -> bool:
    return method.upper() in _MUTATING and path.startswith("/api/")


# ── listener exposure ────────────────────────────────────────────────────────
# IBVAP runs as one local product: the API, dashboard and streams listen on
# loopback only. The single deliberate exception is phone camera intake during
# testing/demos — network.lan_phone_intake opens the HTTPS listener to the LAN,
# and ListenerGuard then serves nothing on it except the phone page and its
# frame WebSocket.
PHONE_INTAKE_PREFIXES = ("/cam/", "/camera/", "/ws/camera/")


def lan_listener_allows(path: str) -> bool:
    return path.startswith(PHONE_INTAKE_PREFIXES)


class ListenerGuard:
    """ASGI middleware: requests arriving on ``lan_port`` may only reach the
    phone intake routes. A plain HTTP middleware would not see WebSockets."""

    def __init__(self, app, lan_port: int | None = None) -> None:
        self.app = app
        self.lan_port = lan_port

    async def __call__(self, scope, receive, send):
        if (self.lan_port is not None
                and scope.get("type") in ("http", "websocket")
                and (scope.get("server") or (None, None))[1] == self.lan_port
                and not lan_listener_allows(scope.get("path", ""))):
            if scope["type"] == "http":
                body = b'{"error":"not available on the camera intake listener"}'
                await send({"type": "http.response.start", "status": 404,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
            else:
                await send({"type": "websocket.close", "code": 1008})
            return
        await self.app(scope, receive, send)


def is_authorized(method: str, path: str, client_host: str | None,
                  header_token: str | None, token: str,
                  allow_loopback: bool = True,
                  require_token: bool = True) -> bool:
    if not require_token or not needs_token(method, path):
        return True
    if allow_loopback and is_loopback(client_host):
        return True
    if not token or not header_token:
        return False
    return hmac.compare_digest(header_token.strip(), token)
