"""
alert_forward.py — deliver events off the machine.

`EventStore` has shipped a `synced` column, an `ix_events_unsynced` index,
`unsynced()` and `mark_synced()` since the beginning, and nothing ever called
them: a store-and-forward queue built for a forwarder that did not exist. Every
row sat unsynced forever. This is the forwarder.

Design notes that matter at a remote Border Out Post:

* **The uplink is assumed absent.** Hours offline is the normal case, not the
  error case. The queue is SQLite, so it already survives a power cut; the
  forwarder's job is to drain it in order when the link returns and to stay
  cheap while it cannot.
* **In order, at least once, per sink.** A C2 operator correlating a breach
  with the follow-up that confirms it needs them in sequence, so a failure
  holds that sink's cursor rather than skipping past it. Each sink keeps its
  own cursor, so one dead endpoint cannot stall a healthy one.
* **Never on the inference thread.** Its own thread and its own SQLite
  connection; a five-second webhook timeout must not cost a frame.
* **Backoff with jitter.** Several posts reconnecting to the same HQ after a
  regional outage should not arrive as a thundering herd.

An event is marked synced once every *enabled* sink has taken it (or when no
sink is configured at all, so the queue cannot grow without bound on a
standalone install).
"""
from __future__ import annotations

import logging
import random
import threading
import time

from .sinks import Sink, build_sinks

logger = logging.getLogger("ibvap.alerts")


class _SinkState:
    def __init__(self, sink: Sink) -> None:
        self.sink = sink
        self.cursor: str | None = None      # last event_id this sink accepted
        self.failures = 0
        self.retry_after = 0.0
        self.sent = 0
        self.dropped = 0
        self.last_ok: float | None = None
        self.last_error = ""

    def backoff(self, base: float, cap: float) -> None:
        self.failures += 1
        delay = min(cap, base * (2 ** min(self.failures - 1, 10)))
        self.retry_after = time.monotonic() + delay * (0.5 + random.random())

    def recovered(self) -> None:
        self.failures = 0
        self.retry_after = 0.0
        self.last_ok = time.time()
        self.last_error = ""


class AlertForwarder:
    """Drains EventStore.unsynced() into the configured sinks."""

    def __init__(self, store, cfg: dict | None = None) -> None:
        c = cfg or {}
        self.store = store
        self.enabled = bool(c.get("enabled", True))
        self.poll_s = float(c.get("poll_interval_s", 1.0))
        self.batch = int(c.get("batch", 100))
        self.backoff_base_s = float(c.get("backoff_base_s", 2.0))
        self.backoff_max_s = float(c.get("backoff_max_s", 300.0))
        self.max_attempts = int(c.get("max_attempts", 0))   # 0 = retry forever
        self.retention_days = float(c.get("retention_days", 7.0))
        self.max_rows = int(c.get("max_queue_rows", 200_000))
        self._sinks = [_SinkState(s) for s in build_sinks(c.get("sinks") or [])]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_prune = 0.0
        self._delivered = 0

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if not self.enabled:
            logger.info("Alert forwarding disabled")
            return
        live = [s for s in self._sinks if s.sink.enabled]
        if not live:
            logger.info("Alert forwarding on, but no sink is enabled — events "
                        "are logged locally only")
        else:
            logger.info("Alert forwarding ready — %d sink(s): %s",
                        len(live), ", ".join(s.sink.label for s in live))
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="alerts")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # ── the loop ────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._drain_once()
            except Exception as e:                   # never let the thread die
                logger.warning("alert forwarder: %s", e)
            self._stop.wait(self.poll_s)

    def _drain_once(self) -> None:
        now = time.monotonic()
        rows = self.store.unsynced(limit=self.batch)
        if rows:
            live = [s for s in self._sinks if s.sink.enabled]
            # Each sink advances its own cursor as far as it can. A sink that
            # fails stops taking anything further *this pass* — its own stream
            # must stay in order — but it does not hold up the others, so one
            # unreachable endpoint cannot blind a working one.
            blocked: set[int] = set()
            for row in rows:
                for st in live:
                    if id(st) not in blocked:
                        self._offer(row, st, now, blocked)
            # An event leaves the queue only once every sink has passed it, and
            # only if every earlier event has too — so a recovering sink
            # replays from exactly where it stopped, with no hole.
            for row in rows:
                if not all(st.cursor is not None and row["event_id"] <= st.cursor
                           for st in live):
                    break
                self.store.mark_synced(row["event_id"] or row["id"])
                self._delivered += 1
        # Retention runs regardless — a post with no sinks still must not fill
        # its disk, and that is also where the honest "offline buffer" number
        # in the docs comes from.
        if time.monotonic() - self._last_prune > 300.0:
            self._last_prune = time.monotonic()
            n = self.store.prune(self.retention_days, self.max_rows)
            if n:
                logger.info("Pruned %d delivered event(s) past retention", n)

    def _offer(self, row: dict, st: _SinkState, now: float,
               blocked: set[int]) -> None:
        """Offer one event to one sink. On failure the sink is marked blocked
        for the rest of this pass so its own delivery stays ordered."""
        if st.cursor is not None and row["event_id"] <= st.cursor:
            return                                   # this sink already has it
        if not st.sink.wants(row):
            st.cursor = row["event_id"]              # filtered out, not failed
            return
        if st.retry_after and now < st.retry_after:
            blocked.add(id(st))                      # still backing off
            return
        ok, detail = st.sink.send(row)
        if ok:
            st.cursor = row["event_id"]
            st.sent += 1
            if st.failures:
                logger.info("Alert sink %s recovered after %d failure(s)",
                            st.sink.label, st.failures)
            st.recovered()
            return
        st.last_error = detail
        st.backoff(self.backoff_base_s, self.backoff_max_s)
        if st.failures == 1 or st.failures % 20 == 0:
            logger.warning("Alert sink %s failed (%d): %s",
                           st.sink.label, st.failures, detail)
        if self.max_attempts and st.failures >= self.max_attempts:
            # Give up on this one event rather than block the queue behind a
            # permanently poisoned record. Only when explicitly allowed.
            logger.error("Alert sink %s giving up on %s after %d attempts",
                         st.sink.label, row.get("event_id"), st.failures)
            st.cursor = row["event_id"]
            st.dropped += 1
            st.recovered()
            return
        blocked.add(id(st))

    # ── observability ───────────────────────────────────────────────────────
    def status(self) -> dict:
        try:
            pending = self.store.unsynced_count()
        except Exception:
            pending = -1
        now = time.monotonic()
        return {
            "enabled": self.enabled,
            "pending": pending,
            "delivered": self._delivered,
            "retention_days": self.retention_days,
            "sinks": [
                {
                    **st.sink.describe(),
                    "sent": st.sent,
                    "dropped": st.dropped,
                    "failures": st.failures,
                    "last_ok": st.last_ok,
                    "last_error": st.last_error,
                    "backing_off_for": (round(max(0.0, st.retry_after - now), 1)
                                        if st.retry_after else 0.0),
                }
                for st in self._sinks
            ],
        }
