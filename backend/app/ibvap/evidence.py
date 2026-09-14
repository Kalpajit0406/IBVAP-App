from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


class EvidenceChain:
    """
    Append-only SHA-256 hash chain stored as newline-delimited JSON.
    Each record includes the hash of the previous record, making the log
    tamper-evident: any modification breaks every subsequent hash.
    """

    GENESIS = "0" * 64

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Serialises append() — the prev-hash read/write must be atomic per
        # record or the chain forks. Risk-level events and confirmed plates are
        # both appended from the inference worker; the lock keeps it correct if
        # another producer is ever added.
        self._lock = threading.Lock()
        self._prev_hash = self._load_tip()

    def _load_tip(self) -> str:
        if not self._path.exists():
            return self.GENESIS
        data = self._path.read_bytes()
        if data and not data.endswith(b"\n"):
            # Power loss mid-append left a fragment. Terminate it rather than
            # delete it: the next record must not be glued onto it, and the
            # damage has to stay visible to verify() instead of vanishing.
            with self._path.open("ab") as f:
                f.write(b"\n")
        for raw in reversed(self._complete_lines()):
            try:
                return json.loads(raw).get("hash", self.GENESIS)
            except ValueError:
                continue
        return self.GENESIS

    def append(self, event: dict) -> str:
        """Append an event and return its SHA-256 hash."""
        with self._lock:
            record = {
                "timestamp": time.time(),
                "prev_hash": self._prev_hash,
                "event": event,
            }
            payload = json.dumps(record, sort_keys=True).encode()
            record["hash"] = hashlib.sha256(payload).hexdigest()

            with self._path.open("a") as f:
                f.write(json.dumps(record) + "\n")

            self._prev_hash = record["hash"]
            return record["hash"]

    def _complete_lines(self) -> list[bytes]:
        # A record still being written by append() on another thread has no
        # trailing newline yet; judging that fragment would report tampering on
        # a perfectly healthy live chain, so only newline-terminated lines count.
        data = self._path.read_bytes()
        if not data.endswith(b"\n"):
            data = data[: data.rfind(b"\n") + 1]
        return [ln for ln in data.splitlines() if ln.strip()]

    def verify(self) -> tuple[bool, int]:
        """
        Walk the chain from genesis. Returns (ok, first_bad_line_number).
        ok=True means the chain is intact. A malformed line counts as tampering
        rather than raising, so a damaged file reports where it broke.
        """
        if not self._path.exists():
            return True, -1

        prev = self.GENESIS
        for lineno, raw in enumerate(self._complete_lines(), start=1):
            try:
                rec = json.loads(raw)
            except ValueError:
                return False, lineno
            if not isinstance(rec, dict) or rec.get("prev_hash") != prev:
                return False, lineno
            check = {k: v for k, v in rec.items() if k != "hash"}
            expected = hashlib.sha256(
                json.dumps(check, sort_keys=True).encode()
            ).hexdigest()
            if rec.get("hash") != expected:
                return False, lineno
            prev = rec["hash"]
        return True, -1

    def summary(self) -> dict:
        """Integrity verdict plus size and tip, for the operator console."""
        if not self._path.exists():
            return {"ok": True, "bad_line": -1, "records": 0,
                    "tip": self.GENESIS, "path": str(self._path)}
        ok, bad = self.verify()
        lines = self._complete_lines()
        tip = self.GENESIS
        if lines:
            try:
                tip = json.loads(lines[-1]).get("hash", self.GENESIS)
            except ValueError:
                pass
        return {"ok": ok, "bad_line": bad, "records": len(lines),
                "tip": tip, "path": str(self._path)}

    def tail(self, limit: int = 50) -> list[dict]:
        """Newest-first complete records; unparseable lines are skipped."""
        if not self._path.exists():
            return []
        out = []
        for raw in reversed(self._complete_lines()[-max(1, int(limit)):]):
            try:
                out.append(json.loads(raw))
            except ValueError:
                continue
        return out
