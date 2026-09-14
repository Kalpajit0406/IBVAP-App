"""
test_security.py -- write-access policy for the HTTP API (ibvap/security.py).
No network.

    python tests/test_security.py
    pytest  tests/test_security.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio                                                    # noqa: E402

from ibvap.security import (ListenerGuard, is_authorized,          # noqa: E402
                            is_loopback, lan_listener_allows,
                            load_or_create_token, needs_token)

CASES = []
TOK = "t" * 43


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("reads are always allowed, even from the LAN with no token")
def _():
    assert is_authorized("GET", "/api/streams", "10.0.0.9", None, TOK)
    assert is_authorized("GET", "/status", "10.0.0.9", None, TOK)


@case("non-/api/ POSTs are outside the policy")
def _():
    assert not needs_token("POST", "/ws/camera/0")
    assert needs_token("DELETE", "/api/fences/f1")


@case("remote write without a token is refused")
def _():
    assert not is_authorized("POST", "/api/shutdown", "192.168.1.20", None, TOK)
    assert not is_authorized("DELETE", "/api/streams/2", "192.168.1.20", "", TOK)


@case("remote write with the wrong token is refused, right token allowed")
def _():
    assert not is_authorized("POST", "/api/shutdown", "192.168.1.20", "nope", TOK)
    assert is_authorized("POST", "/api/shutdown", "192.168.1.20", TOK, TOK)


@case("loopback writes are allowed without a token (IPv4 and IPv6)")
def _():
    assert is_authorized("POST", "/api/layout", "127.0.0.1", None, TOK)
    assert is_authorized("POST", "/api/layout", "::1", None, TOK)
    assert not is_authorized("POST", "/api/layout", "127.0.0.1", None, TOK,
                             allow_loopback=False)


@case("is_loopback rejects junk and LAN addresses")
def _():
    assert not is_loopback(None) and not is_loopback("")
    assert not is_loopback("10.1.1.1") and not is_loopback("evil")
    assert is_loopback("127.5.5.5")


@case("token file is created once and then reused")
def _():
    p = Path(tempfile.mkdtemp()) / "sub" / "api_token"
    a = load_or_create_token(p)
    b = load_or_create_token(p)
    assert a == b and len(a) >= 24 and p.is_file()


@case("a too-short token file is replaced with a strong one")
def _():
    p = Path(tempfile.mkdtemp()) / "api_token"
    p.write_text("abc")
    t = load_or_create_token(p)
    assert len(t) >= 24 and p.read_text().strip() == t


@case("require_token=False lets remote writes through (testing mode)")
def _():
    assert is_authorized("POST", "/api/shutdown", "192.168.1.20", None, TOK,
                         require_token=False)


def _drive(guard, scope):
    sent, reached = [], []

    async def inner(scope, receive, send):
        reached.append(scope["path"])

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "websocket.connect"}

    guard.app = inner
    asyncio.run(guard(scope, receive, send))
    return sent, reached


@case("only phone intake paths are allowed on the LAN listener")
def _():
    assert lan_listener_allows("/cam/0") and lan_listener_allows("/camera/3")
    assert lan_listener_allows("/ws/camera/1")
    for p in ("/status", "/api/shutdown", "/stream", "/monitor", "/", "/snap/0/a.jpg"):
        assert not lan_listener_allows(p), p


@case("guard blocks API/stream HTTP on the LAN port, passes the phone page")
def _():
    g = ListenerGuard(None, lan_port=8443)
    sent, reached = _drive(g, {"type": "http", "path": "/api/streams",
                               "server": ("192.168.1.9", 8443)})
    assert not reached and sent[0]["status"] == 404
    sent, reached = _drive(g, {"type": "http", "path": "/cam/0",
                               "server": ("192.168.1.9", 8443)})
    assert reached == ["/cam/0"] and not sent


@case("guard closes non-intake WebSockets on the LAN port")
def _():
    g = ListenerGuard(None, lan_port=8443)
    sent, reached = _drive(g, {"type": "websocket", "path": "/ws/other",
                               "server": ("192.168.1.9", 8443)})
    assert not reached and sent == [{"type": "websocket.close", "code": 1008}]
    sent, reached = _drive(g, {"type": "websocket", "path": "/ws/camera/0",
                               "server": ("192.168.1.9", 8443)})
    assert reached == ["/ws/camera/0"]


@case("guard leaves the local API listener and lifespan untouched")
def _():
    g = ListenerGuard(None, lan_port=8443)
    _, reached = _drive(g, {"type": "http", "path": "/api/shutdown",
                            "server": ("127.0.0.1", 8090)})
    assert reached == ["/api/shutdown"]
    _, reached = _drive(ListenerGuard(None, lan_port=None),
                        {"type": "http", "path": "/status", "server": ("x", 8443)})
    assert reached == ["/status"]


def run() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  -- {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
