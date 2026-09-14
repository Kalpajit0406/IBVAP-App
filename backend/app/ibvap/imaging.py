"""
imaging.py — tiny shared JPEG-encode / filename-slug helpers used by every
evidence writer (ibvap/snapshots.py, ibvap/anpr_events.py) so the encode +
SHA-256 logic and the filesystem-safe-fragment rule live in exactly one place.
"""
from __future__ import annotations

import hashlib
import re

import cv2

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def encode_jpeg(frame, quality: int = 90) -> tuple[bytes, str] | None:
    """JPEG-encode a BGR frame, returning (bytes, sha256_hex) or None if the
    frame is empty or encoding fails."""
    if frame is None:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    except cv2.error:
        return None
    if not ok:
        return None
    data = buf.tobytes()
    return data, hashlib.sha256(data).hexdigest()


def slug(s: str, maxlen: int = 28) -> str:
    """A filesystem-safe fragment for a snapshot/event filename."""
    s = _SLUG_RE.sub("-", str(s or "")).strip("-")
    return s[:maxlen]
