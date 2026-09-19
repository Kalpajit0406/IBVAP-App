from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2

# ── Severity and category ───────────────────────────────────────────────────
# These were one column. `level` was written as "Critical"/"High" (risk edges)
# but also as "Breach", "Posture" and "Plate" — severity and category squashed
# together, so counts_by_level() was summing apples and oranges and no consumer
# could filter "everything Critical" or "every intrusion". They are separate
# now; `level` is still written, unchanged, for anything reading the old column.
SEVERITIES = ("Info", "Low", "Medium", "High", "Critical")
CATEGORIES = ("risk", "intrusion", "loiter", "crossing", "running", "group",
              "following", "weapon", "posture", "face_match", "plate")

# Reproduces the exact legacy `level` value each call site used to pass, so the
# old column keeps its historical meaning rather than silently changing.
_LEGACY_LEVEL = {"intrusion": "Breach", "posture": "Posture", "plate": "Plate"}

# Crockford base32 — ULID's alphabet (no I, L, O, U: unambiguous when read aloud
# over a radio, which is not a hypothetical requirement at a border post).
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


_mono_lock = threading.Lock()
_last_ms = 0
_last_rand = 0
_RAND_MAX = 1 << 80


def new_event_id(ts: float | None = None) -> str:
    """A ULID: 48-bit millisecond timestamp + 80 bits of randomness, base32.

    Chosen over UUID4 because it sorts lexicographically by creation time, so
    the same value serves as the event's global identity *and* as the cursor a
    command-and-control system pages the feed with. A per-install autoincrement
    id could do neither — two Border Out Posts would both emit id=1.

    Live ids are **strictly monotonic**, which is not decoration. Two events in
    the same millisecond with independently random suffixes can come out in
    descending order; an event written after a consumer had already paged past
    that point would then sort *before* its cursor and never be delivered. The
    ULID spec's monotonicity rule — reuse the millisecond and increment the
    random field — removes that whole class of silent loss.

    Passing an explicit `ts` (only the schema migration does) skips the
    monotonic state deliberately: those ids describe history and must keep
    their own timestamps rather than being dragged up to now.
    """
    global _last_ms, _last_rand
    if ts is not None:
        ms = int(ts * 1000)
        rand = int.from_bytes(secrets.token_bytes(10), "big")
    else:
        with _mono_lock:
            ms = int(time.time() * 1000)
            if ms > _last_ms:
                _last_ms, _last_rand = ms, int.from_bytes(secrets.token_bytes(10), "big")
            else:
                # Same millisecond, or the clock stepped backwards (an NTP
                # correction at a remote post). Never emit a smaller id.
                ms = _last_ms
                _last_rand += 1
                if _last_rand >= _RAND_MAX:          # exhausted this millisecond
                    _last_ms = ms = _last_ms + 1
                    _last_rand = int.from_bytes(secrets.token_bytes(10), "big")
            rand = _last_rand
    n = (ms << 80) | rand
    return "".join(_B32[(n >> (5 * i)) & 31] for i in range(25, -1, -1))


def iso_utc(ts: float) -> str:
    """ISO-8601 in UTC. The bare epoch float a remote post writes is
    unrecoverable once its clock drifts; an explicit zone is not optional for
    evidence that has to correlate across sites."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


class EventStore:
    """
    SQLite-backed store-and-forward event queue.

    All writes are serialised through a lock so the store is safe to call from
    multiple threads. `unsynced()` / `mark_synced()` are drained by
    ibvap/alert_forward.py, which delivers each event to the configured sinks
    and is the reason this is a queue rather than a log.
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS events (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ts        REAL    NOT NULL,
        cam_id    INTEGER NOT NULL,
        level     TEXT    NOT NULL,
        score     REAL    NOT NULL,
        persons   INTEGER NOT NULL DEFAULT 0,
        vehicles  INTEGER NOT NULL DEFAULT 0,
        details   TEXT    NOT NULL DEFAULT '{}',
        ev_hash   TEXT,
        synced    INTEGER NOT NULL DEFAULT 0
    )
    """

    # Added in schema 2. Applied with ALTER TABLE against a live database, so
    # an existing install keeps its history instead of starting over.
    _ADDED = (
        ("event_id", "TEXT"),
        ("ts_utc", "TEXT"),
        ("severity", "TEXT"),
        ("category", "TEXT"),
        ("site_id", "TEXT"),
        ("post_name", "TEXT"),
        ("cam_name", "TEXT"),
        ("lat", "REAL"),
        ("lon", "REAL"),
        ("acked_at", "REAL"),
        ("acked_by", "TEXT"),
        ("ack_note", "TEXT"),
        ("ack_hash", "TEXT"),
        ("schema_version", "INTEGER"),
        ("producer", "TEXT"),
    )

    def __init__(self, db_path: str, site: dict | None = None,
                 producer: str | None = None) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._site = dict(site or {})
        self._cams: dict[int, dict] = {}
        self._producer = producer or f"ibvap@{os.environ.get('COMPUTERNAME', 'unknown')}"
        with self._lock:
            # WAL: a reader (the /status query, the forwarder's drain) never
            # blocks the detection thread's writes, and an unclean shutdown
            # can't corrupt the file — it just replays the log.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(self.DDL)
            self._migrate()
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_events_unsynced "
                "ON events(synced) WHERE synced = 0")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_events_event_id ON events(event_id)")
            self._conn.commit()

    # ── Migration ───────────────────────────────────────────────────────────
    def _migrate(self) -> None:
        have = {r[1] for r in self._conn.execute("PRAGMA table_info(events)")}
        added = [c for c, t in self._ADDED if c not in have]
        for col, typ in self._ADDED:
            if col not in have:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {col} {typ}")
        if not added:
            return
        # Backfill history so old rows are queryable on the new columns rather
        # than being a null hole the feed skips. The legacy `level` carried both
        # meanings, so it can be split back apart without guessing.
        for legacy, sev, cat in (("Breach", "Critical", "intrusion"),
                                 ("Posture", "High", "posture"),
                                 ("Plate", "Info", "plate")):
            self._conn.execute(
                "UPDATE events SET severity=?, category=? "
                "WHERE severity IS NULL AND level=?", (sev, cat, legacy))
        # Anything left held a real severity already; that was the risk edge.
        self._conn.execute(
            "UPDATE events SET severity=level, category='risk' WHERE severity IS NULL")
        self._conn.execute(
            "UPDATE events SET schema_version=? WHERE schema_version IS NULL",
            (SCHEMA_VERSION,))
        # Backfilled rows get an id derived from their own timestamp, so the
        # ULID ordering stays consistent with the history it describes.
        rows = self._conn.execute(
            "SELECT id, ts FROM events WHERE event_id IS NULL").fetchall()
        for rid, ts in rows:
            self._conn.execute(
                "UPDATE events SET event_id=?, ts_utc=? WHERE id=?",
                (new_event_id(ts), iso_utc(ts), rid))
        self._conn.commit()

    # ── Identity ────────────────────────────────────────────────────────────
    def set_site(self, site: dict) -> None:
        """Site/post identity, stamped on every event so two posts' logs can be
        merged by whoever receives them."""
        self._site = dict(site or {})

    def set_cameras(self, cams: dict) -> None:
        """{cam_id: {"name": str, "lat": float|None, "lon": float|None}}"""
        self._cams = {int(k): dict(v) for k, v in (cams or {}).items()}

    # ── Write ───────────────────────────────────────────────────────────────
    def log(
        self,
        cam_id: int,
        severity: str,
        score: float,
        persons: int = 0,
        vehicles: int = 0,
        details: dict | None = None,
        ev_hash: str | None = None,
        category: str = "risk",
    ) -> str:
        """Record one event. Returns its ULID `event_id`."""
        # No explicit ts: that argument is the migration's historical path and
        # skips monotonicity, which the live feed's cursor depends on.
        eid = new_event_id()
        ts = time.time()
        cam = self._cams.get(int(cam_id), {})
        level = _LEGACY_LEVEL.get(category, severity)
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (event_id, ts, ts_utc, cam_id, cam_name, lat, lon,"
                " site_id, post_name, level, severity, category, score, persons,"
                " vehicles, details, ev_hash, schema_version, producer)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    eid, ts, iso_utc(ts), int(cam_id), cam.get("name"),
                    cam.get("lat"), cam.get("lon"),
                    self._site.get("site_id"), self._site.get("post_name"),
                    level, severity, category, score, persons, vehicles,
                    json.dumps(details or {}), ev_hash, SCHEMA_VERSION,
                    self._producer,
                ),
            )
            self._conn.commit()
        return eid

    def ack(self, event_id: str, operator: str, note: str = "") -> dict | None:
        """Persist an acknowledgement. Returns the updated row, or None if the
        event is unknown. The caller appends it to the evidence chain and passes
        the hash back to :meth:`set_ack_hash` — acknowledging an alert is itself
        chain-of-custody, not just UI state."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE events SET acked_at=?, acked_by=?, ack_note=? "
                "WHERE event_id=? AND acked_at IS NULL",
                (now, str(operator)[:120], str(note)[:500], str(event_id)))
            self._conn.commit()
            if cur.rowcount == 0:
                # Either unknown, or already acknowledged — an ack is not
                # re-writable, so the first one stands.
                self._conn.row_factory = sqlite3.Row
                row = self._conn.execute(
                    "SELECT * FROM events WHERE event_id=?", (str(event_id),)).fetchone()
                return dict(row) if row else None
            self._conn.row_factory = sqlite3.Row
            row = self._conn.execute(
                "SELECT * FROM events WHERE event_id=?", (str(event_id),)).fetchone()
        return dict(row) if row else None

    def set_ack_hash(self, event_id: str, ack_hash: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET ack_hash=? WHERE event_id=?",
                               (ack_hash, str(event_id)))
            self._conn.commit()

    # ── Forwarder queue ─────────────────────────────────────────────────────
    def unsynced(self, limit: int = 200) -> list[dict]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(
                "SELECT * FROM events WHERE synced=0 ORDER BY id LIMIT ?",
                (int(limit),)).fetchall()
        return [self._row(r) for r in rows]

    def mark_synced(self, event_id) -> None:
        """Accepts a ULID event_id or a legacy integer rowid."""
        col = "id" if isinstance(event_id, int) else "event_id"
        with self._lock:
            self._conn.execute(
                f"UPDATE events SET synced=1 WHERE {col}=?", (event_id,))
            self._conn.commit()

    def unsynced_count(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE synced=0").fetchone()[0]

    def prune(self, max_age_days: float = 30.0, max_rows: int = 200_000) -> int:
        """Bound the queue. Unbounded growth is how an unattended post fills its
        disk; a synced row older than the retention has already been delivered
        and is no longer needed for replay."""
        cutoff = time.time() - max_age_days * 86400.0
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM events WHERE synced=1 AND ts < ?", (cutoff,))
            n = cur.rowcount or 0
            over = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            if over > max_rows:
                cur = self._conn.execute(
                    "DELETE FROM events WHERE id IN (SELECT id FROM events "
                    "WHERE synced=1 ORDER BY id LIMIT ?)", (over - max_rows,))
                n += cur.rowcount or 0
            self._conn.commit()
        return n

    # ── Read ────────────────────────────────────────────────────────────────
    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        try:
            d["details"] = json.loads(d.get("details") or "{}")
        except ValueError:
            d["details"] = {}
        return d

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def query(self, limit: int = 100, cam_id: int | None = None,
              level: str | None = None, since: float | None = None,
              severity: str | None = None, category: str | None = None,
              since_id: str | None = None, until: float | None = None,
              ascending: bool = False) -> list[dict]:
        """Incident rows as plain dicts (details decoded).

        `since_id` is the ULID cursor a C2 client pages with: pass the last
        event_id you saw and get everything after it, in order. That is what
        makes the feed resumable across a restart or a dropped connection,
        which a timestamp filter alone cannot guarantee.
        """
        limit = max(1, min(int(limit), 1000))
        sql = "SELECT * FROM events"
        where, args = [], []
        if cam_id is not None:
            where.append("cam_id = ?"); args.append(int(cam_id))
        if level:
            where.append("level = ?"); args.append(str(level))
        if severity:
            where.append("severity = ?"); args.append(str(severity))
        if category:
            where.append("category = ?"); args.append(str(category))
        if since is not None:
            where.append("ts >= ?"); args.append(float(since))
        if until is not None:
            where.append("ts <= ?"); args.append(float(until))
        if since_id:
            where.append("event_id > ?"); args.append(str(since_id))
            ascending = True          # a cursor only makes sense moving forward
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY event_id {'ASC' if ascending else 'DESC'} LIMIT ?"
        args.append(limit)
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row(r) for r in rows]

    def counts_by_level(self, since: float | None = None) -> dict[str, int]:
        sql = "SELECT level, COUNT(*) FROM events"
        args: list = []
        if since is not None:
            sql += " WHERE ts >= ?"; args.append(float(since))
        sql += " GROUP BY level"
        with self._lock:
            self._conn.row_factory = None
            return {lvl: n for lvl, n in self._conn.execute(sql, args).fetchall()}

    def counts(self, since: float | None = None) -> dict:
        """Severity and category tallies, separately — the thing
        counts_by_level() could not answer while the two were one column."""
        out = {"severity": {}, "category": {}}
        for field in ("severity", "category"):
            sql = f"SELECT {field}, COUNT(*) FROM events"
            args: list = []
            if since is not None:
                sql += " WHERE ts >= ?"; args.append(float(since))
            sql += f" GROUP BY {field}"
            with self._lock:
                self._conn.row_factory = None
                out[field] = {k or "?": n
                              for k, n in self._conn.execute(sql, args).fetchall()}
        return out

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._conn.close()
