from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class EventStore:
    """
    SQLite-backed store-and-forward event queue.

    All writes are serialised through a lock so the store is safe to call
    from multiple threads. When network connectivity is restored a sync
    routine can drain unsynced rows.
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

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            # WAL: a reader (the /status query, a sync drain) never blocks the
            # detection thread's writes, and an unclean shutdown can't corrupt
            # the file — it just replays the log. synchronous=NORMAL is the
            # right durability/speed trade for an at-least-once event queue.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(self.DDL)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_events_unsynced "
                "ON events(synced) WHERE synced = 0")
            self._conn.commit()

    def log(
        self,
        cam_id: int,
        level: str,
        score: float,
        persons: int = 0,
        vehicles: int = 0,
        details: dict | None = None,
        ev_hash: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (ts, cam_id, level, score, persons, vehicles, details, ev_hash)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    time.time(), cam_id, level, score, persons, vehicles,
                    json.dumps(details or {}), ev_hash,
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def unsynced(self) -> list[sqlite3.Row]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            return self._conn.execute(
                "SELECT * FROM events WHERE synced=0 ORDER BY id"
            ).fetchall()

    def mark_synced(self, event_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET synced=1 WHERE id=?", (event_id,))
            self._conn.commit()

    def recent(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            return self._conn.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()

    def query(self, limit: int = 100, cam_id: int | None = None,
              level: str | None = None, since: float | None = None) -> list[dict]:
        """Newest-first incident rows as plain dicts (details decoded)."""
        limit = max(1, min(int(limit), 1000))
        sql = "SELECT id, ts, cam_id, level, score, persons, vehicles, details, ev_hash FROM events"
        where, args = [], []
        if cam_id is not None:
            where.append("cam_id = ?"); args.append(int(cam_id))
        if level:
            where.append("level = ?"); args.append(str(level))
        if since is not None:
            where.append("ts >= ?"); args.append(float(since))
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            cur = self._conn.execute(sql, args)
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            try:
                d["details"] = json.loads(d["details"] or "{}")
            except ValueError:
                d["details"] = {}
            out.append(d)
        return out

    def counts_by_level(self, since: float | None = None) -> dict[str, int]:
        sql = "SELECT level, COUNT(*) FROM events"
        args: list = []
        if since is not None:
            sql += " WHERE ts >= ?"; args.append(float(since))
        sql += " GROUP BY level"
        with self._lock:
            return {lvl: n for lvl, n in self._conn.execute(sql, args).fetchall()}

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._conn.close()
