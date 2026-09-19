"""
youtube_capture.py — treat a YouTube link as a camera.

Paste a watch URL into the camera form and it becomes an ordinary stream: a
tile on the dashboard with boxes drawn on it, running through detection, ANPR,
face recognition, the fences and the behaviour rules like any other camera.
Nothing downstream knows the difference, because `YouTubeCapture` is an
`RtspCapture` — the muxer, the risk engine and the console all see the same
duck type (see rtsp_capture.py's header).

The only thing this class adds is finding out what to actually open, which is
less obvious than it sounds:

  * **YouTube serves no muxed streams any more.** Every format is video-only,
    either a direct `https` MP4 or an HLS playlist. That is fine here — the
    pipeline has never had audio — but it means "just open the watch URL" was
    never going to work, and neither does yt-dlp's `best` selector (it looks
    for a stream carrying both tracks and finds none).
  * **The media URL expires and is bound to the IP that asked for it** — about
    six hours, stamped into the link itself. So it is re-resolved on expiry,
    and never retried after a failed open: a stale googlevideo link fails the
    same way forever, and the reconnect loop would spin on it.
  * **Recordings need pacing.** A googlevideo URL demuxes at ~470 fps if you
    let it. `RtspCapture._Pacer` holds it to the video's own frame rate; see
    the note there for why this matters beyond the picture looking right.

Format choice, in order of preference: H.264 (OpenCV's bundled FFmpeg decodes
it fastest and most reliably), the tallest that fits `max_height`, and a direct
`https` URL for a recording — one connection, seekable, so rewinding at the end
is a single `CAP_PROP_POS_FRAMES` call. A live stream is HLS by necessity.
DASH segment manifests are rejected outright: OpenCV cannot open one, and
picking one silently is the likeliest way this fails in front of an audience.

Requires `yt-dlp` (in requirements.txt). It is imported lazily, so a missing or
broken install disables this one camera with a readable error instead of taking
the server down — and it does go stale, because YouTube changes: if every link
suddenly fails, update it before looking anywhere else.

This is the one camera kind that needs the internet. The traffic is outbound
only and opens no inbound port, so the product's local-only posture is intact.
"""
from __future__ import annotations

import logging
import re
import time
import urllib.parse

from .rtsp_capture import RtspCapture

logger = logging.getLogger("ibvap.youtube")

# Hosts whose links this class can open. A bare video id is deliberately NOT
# accepted — "0"/"1" already mean a webcam index, and guessing would be worse
# than telling the operator to paste the whole link.
_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "www.youtu.be",
    "youtube-nocookie.com", "www.youtube-nocookie.com",
}

# /expire/1789861988/ei/... — HLS manifests carry the deadline in the path,
# progressive URLs carry it as ?expire=. Both appear in practice.
_PATH_EXPIRE = re.compile(r"/expire/(\d{9,12})(?:/|$)")

_ID_PATHS = ("/shorts/", "/embed/", "/live/", "/v/")


def is_youtube_url(url: str) -> bool:
    """True for a link this class should handle."""
    try:
        parts = urllib.parse.urlsplit(str(url).strip())
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    return parts.netloc.lower().split(":")[0] in _HOSTS


def video_id(url: str) -> str:
    """The 11-character video id, or "" — for display only, never for opening."""
    try:
        parts = urllib.parse.urlsplit(str(url).strip())
    except ValueError:
        return ""
    host = parts.netloc.lower().split(":")[0]
    if host in ("youtu.be", "www.youtu.be"):
        return parts.path.lstrip("/").split("/")[0][:16]
    for prefix in _ID_PATHS:
        if parts.path.startswith(prefix):
            return parts.path[len(prefix):].split("/")[0][:16]
    try:
        return urllib.parse.parse_qs(parts.query).get("v", [""])[0][:16]
    except ValueError:
        return ""


def expiry_of(url: str) -> float:
    """Unix time the media URL stops working; 0.0 when it does not say."""
    try:
        parts = urllib.parse.urlsplit(str(url))
    except ValueError:
        return 0.0
    m = _PATH_EXPIRE.search(parts.path)
    if m:
        return float(m.group(1))
    try:
        got = urllib.parse.parse_qs(parts.query).get("expire", [])
    except ValueError:
        return 0.0
    try:
        return float(got[0]) if got else 0.0
    except (TypeError, ValueError):
        return 0.0


def _codec_rank(vcodec: str | None) -> int:
    v = (vcodec or "").lower()
    if v.startswith(("avc1", "h264")):
        return 3          # H.264 — what the bundled FFmpeg decodes best
    if v.startswith(("vp9", "vp09")):
        return 2
    if v.startswith("av01"):
        return 1          # decodes, but slowly in software
    return 0


def pick_format(info: dict, *, max_height: int = 720) -> dict:
    """Choose the stream to open from a yt-dlp info dict.

    Pure, so the choice is testable against a canned format list without
    touching the network. Raises RuntimeError when nothing is usable.
    """
    live = bool(info.get("is_live"))
    scored = []
    for f in info.get("formats") or []:
        vcodec = f.get("vcodec")
        if not vcodec or vcodec == "none":
            continue                                   # audio-only
        if not f.get("url"):
            continue
        proto = str(f.get("protocol") or "")
        if proto.startswith("http_dash_segments"):
            continue                                   # OpenCV cannot open one
        is_hls = proto.startswith("m3u8")
        if live and not is_hls:
            continue          # a live "https" entry is a placeholder, not media
        if not (is_hls or proto.startswith("http")):
            continue                                   # ftp, ws, whatever else
        height = int(f.get("height") or 0)
        scored.append((
            1 if (height and height <= max_height) else 0,   # fits the cap
            height if (height and height <= max_height) else -height,
            _codec_rank(vcodec),
            # A recording is better served by one seekable file; a live stream
            # has to be HLS and this term never separates its candidates.
            0 if is_hls else 1,
            f,
        ))
    if not scored:
        raise RuntimeError(
            "no playable video stream — the video may be age-restricted, "
            "members-only, region-blocked or DRM-protected"
            if info.get("formats") else
            "YouTube returned no formats for that link")
    scored.sort(key=lambda s: s[:4], reverse=True)
    return scored[0][4]


class YouTubeCapture(RtspCapture):
    """A YouTube link, pulled like a CCTV stream."""

    def __init__(self, cam_id: int, url: str, *,
                 max_height: int = 720,
                 loop_vod: bool = True,
                 refresh_margin_s: float = 300.0,
                 resolve_timeout_s: float = 20.0,
                 **kw) -> None:
        super().__init__(cam_id, url, **kw)
        self.kind = "youtube"
        self.watch_url = str(url).strip()
        self.video_id = video_id(self.watch_url)
        self._max_height = int(max_height)
        self._loop_vod = bool(loop_vod)
        self._refresh_margin = float(refresh_margin_s)
        self._resolve_timeout = float(resolve_timeout_s)

        # Filled in by _resolve(); shown on the console card.
        self.title = ""
        self.is_live = False
        self.quality = ""
        self.duration = 0.0
        self.media_w = self.media_h = 0
        self.media_fps = 0.0
        self.resolved_at = 0.0
        self.resolves = 0
        self.resolve_error = ""
        self._media_url = ""
        self._expires_at = 0.0

    # ── resolution ─────────────────────────────────────────────────────────
    def _stale(self) -> bool:
        return bool(self._expires_at) and \
            time.time() >= self._expires_at - self._refresh_margin

    def _resolve_source(self) -> str:
        if self._media_url and not self._stale():
            return self._media_url
        self._resolve()
        return self._media_url

    def _resolve(self) -> None:
        try:
            from yt_dlp import YoutubeDL           # noqa: PLC0415 — see header
        except Exception as e:
            self.resolve_error = ("yt-dlp is not installed — run "
                                  "`pip install yt-dlp` in the backend runtime")
            raise RuntimeError(self.resolve_error) from e

        opts = {
            "quiet": True, "no_warnings": True, "skip_download": True,
            "noplaylist": True,                    # ...&list=... → just the video
            "socket_timeout": self._resolve_timeout,
        }
        t0 = time.monotonic()
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(self.watch_url, download=False)
        except Exception as e:
            self.resolve_error = _readable(e)
            raise RuntimeError(self.resolve_error) from e
        if info is None:
            self.resolve_error = "YouTube returned nothing for that link"
            raise RuntimeError(self.resolve_error)

        if info.get("_type") == "playlist":        # a channel or playlist link
            entries = [e for e in (info.get("entries") or []) if e]
            if not entries:
                self.resolve_error = "that link has no playable video"
                raise RuntimeError(self.resolve_error)
            info = entries[0]

        fmt = pick_format(info, max_height=self._max_height)

        self._media_url = str(fmt["url"])
        self._expires_at = expiry_of(self._media_url)
        self.title = str(info.get("title") or "")[:120]
        self.is_live = bool(info.get("is_live"))
        self.duration = float(info.get("duration") or 0.0)
        self.media_w = int(fmt.get("width") or 0)
        self.media_h = int(fmt.get("height") or 0)
        self.media_fps = float(fmt.get("fps") or 0.0)
        self.quality = "%dp%s" % (self.media_h, _fps_suffix(self.media_fps))
        self.resolved_at = time.time()
        self.resolves += 1
        self.resolve_error = ""
        if not self.video_id:
            self.video_id = str(info.get("id") or "")[:16]

        # Both kinds need pacing. A recording is obvious — it demuxes far
        # faster than it plays. A live HLS stream is less so: FFmpeg hands over
        # a whole segment at once, so without pacing the decode throttle (which
        # counts wall-clock seconds) spends its budget on the first frames of
        # each segment and throws the rest away — measured at 0.3 fps out of a
        # 30 fps stream. Paced, wall-clock time tracks the video's own time and
        # the throttle samples it evenly. Only rewinding differs.
        self.paced = True
        self.loop_at_end = (not self.is_live) and self._loop_vod

        logger.info("%s YouTube %s — %r (%s %s%s) in %.1fs",
                    "probe" if self.cam_id < 0 else "CAM-%02d" % self.cam_id,
                    self.video_id, self.title, self.quality,
                    "LIVE" if self.is_live else "recorded",
                    "" if self.is_live else (", loops" if self.loop_at_end else ""),
                    time.monotonic() - t0)

    def _open_capture(self):
        cap = super()._open_capture()
        if cap is None:
            # These links are time-limited and bound to the IP that asked for
            # them; retrying the same one just fails again. Drop it so the next
            # attempt resolves afresh.
            self._media_url = ""
        return cap

    # ── dashboard ──────────────────────────────────────────────────────────
    def info(self) -> dict:
        base = super().info()
        base["youtube"] = {
            "title": self.title,
            "video_id": self.video_id,
            "is_live": self.is_live,
            "quality": self.quality,
            "duration_s": round(self.duration, 1),
            "resolved_at": round(self.resolved_at, 1),
            "expires_in_s": (round(self._expires_at - time.time())
                             if self._expires_at else 0),
            "resolves": self.resolves,
            "resolve_error": self.resolve_error,
        }
        return base


def probe(url: str, *, max_height: int = 720, timeout_s: float = 20.0) -> dict:
    """Resolve a link and describe what would play, for the console's Test button.

    Same `{"ok": ...}` shape the RTSP probe returns. Nothing is opened and no
    thread is started — this only asks YouTube what is there, which is the part
    an operator needs to see before saving the camera.
    """
    cap = YouTubeCapture(-1, url, max_height=max_height, resolve_timeout_s=timeout_s)
    t0 = time.monotonic()
    try:
        cap._resolve()
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:                            # pragma: no cover
        return {"ok": False, "error": _readable(e)}
    mins, secs = divmod(int(cap.duration), 60)
    return {
        "ok": True,
        "width": cap.media_w,
        "height": cap.media_h,
        "fps": cap.media_fps or 24.0,
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        "title": cap.title,
        "is_live": cap.is_live,
        "quality": cap.quality,
        "duration_s": round(cap.duration),
        "note": "%s — %s%s" % (cap.title or cap.video_id, cap.quality,
                               " LIVE" if cap.is_live
                               else (" · %d:%02d" % (mins, secs) if cap.duration else "")),
    }


def _fps_suffix(fps) -> str:
    try:
        n = int(round(float(fps or 0)))
    except (TypeError, ValueError):
        return ""
    return str(n) if n > 30 else ""          # "720p60"; plain "720p" otherwise


def _readable(exc: Exception) -> str:
    """yt-dlp errors arrive wrapped in ANSI colour and a 'ERROR: ' prefix."""
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).strip()
    msg = re.sub(r"^ERROR:\s*", "", msg)
    msg = re.sub(r"^\[[^\]]+\]\s*[\w-]{11}:\s*", "", msg)   # "[youtube] <id>: "
    return msg[:300] or exc.__class__.__name__
