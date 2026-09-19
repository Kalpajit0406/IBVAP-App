"""
test_youtube_capture.py -- turning a YouTube link into a camera.

Three things here can break silently and only show up in front of an audience,
so they are pinned down:

  * **Which stream gets opened.** YouTube serves no muxed formats any more —
    everything is video-only, and the list mixes direct MP4s, HLS playlists and
    DASH segment manifests that OpenCV cannot open at all. Picking one of those
    last is a camera that never produces a frame. The choice is a pure function
    of the format list, so it is tested against canned ones.
  * **Pacing.** A recorded source demuxes at ~470 fps if nothing holds it back,
    and a live HLS stream arrives a whole segment at a time. Both need holding
    to the video's own frame rate; `_Pacer` does it, with an injected clock here
    so the tests cost no real time.
  * **Expiry.** The media URL dies after ~6 hours and is bound to this machine's
    IP. If a stale one is retried instead of re-resolved, the camera never comes
    back.

No network: yt-dlp is replaced with a stub, so these run on an air-gapped box
and cannot break because someone deleted a video.

    python tests/test_youtube_capture.py
    pytest  tests/test_youtube_capture.py
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.rtsp_capture import RtspCapture, _Pacer          # noqa: E402
from ibvap.youtube_capture import (YouTubeCapture, expiry_of,  # noqa: E402
                                   is_youtube_url, pick_format, probe,
                                   video_id)

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# ── canned yt-dlp data ──────────────────────────────────────────────────────
def _fmt(fid, height, vcodec, protocol, **kw):
    f = {"format_id": fid, "height": height, "width": height * 16 // 9,
         "vcodec": vcodec, "acodec": "none", "protocol": protocol,
         "url": f"https://rr1.googlevideo.com/videoplayback?expire=2000000000&id={fid}",
         "fps": 30, "ext": "mp4"}
    f.update(kw)
    return f


# Shaped after a real `extract_info` on a 4K 60 fps upload (Sept 2026).
VOD_INFO = {
    "id": "aqz-KE-bpKQ", "title": "A Clip", "is_live": False, "duration": 635,
    "formats": [
        {"format_id": "140", "vcodec": "none", "acodec": "mp4a", "protocol": "https",
         "url": "https://rr1.googlevideo.com/videoplayback?id=140"},   # audio-only
        _fmt("160", 144, "avc1.4d400c", "https"),
        _fmt("134", 360, "avc1.4d401e", "https"),
        _fmt("243", 360, "vp9", "https"),
        _fmt("298", 720, "avc1.4d4020", "https", fps=60),
        _fmt("302", 720, "vp9", "https", fps=60),
        _fmt("398", 720, "av01.0.08M.0", "https", fps=60),
        _fmt("311", 720, "avc1.4D4020", "m3u8_native", fps=60),
        _fmt("299", 1080, "avc1.64002a", "https", fps=60),
        _fmt("616", 1080, "vp09.00.41.0", "http_dash_segments", fps=60),
    ],
}

LIVE_INFO = {
    "id": "CXYr04BWvmc", "title": "Bridge Cam", "is_live": True, "duration": None,
    "formats": [
        _fmt("93", 360, "avc1.4d401e", "m3u8_native"),
        _fmt("95", 720, "avc1.4d401f", "m3u8_native"),
        _fmt("96", 1080, "avc1.64002a", "m3u8_native"),
        # Live entries sometimes carry an https row that is not playable media.
        _fmt("299", 720, "avc1.64002a", "https"),
    ],
}


class _FakeYDL:
    """Stands in for yt_dlp.YoutubeDL. Set the class attributes per test."""
    info: dict | None = None
    error: Exception | None = None
    last_opts: dict = {}

    def __init__(self, opts):
        _FakeYDL.last_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        if _FakeYDL.error is not None:
            raise _FakeYDL.error
        return _FakeYDL.info


@contextmanager
def fake_ytdlp(info=None, error=None):
    """Install the stub for the duration of the block. `_resolve` imports
    yt_dlp inside the function, so swapping sys.modules is enough."""
    _FakeYDL.info, _FakeYDL.error = info, error
    saved = sys.modules.get("yt_dlp")
    sys.modules["yt_dlp"] = types.SimpleNamespace(YoutubeDL=_FakeYDL)
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("yt_dlp", None)
        else:
            sys.modules["yt_dlp"] = saved
        _FakeYDL.info = _FakeYDL.error = None


@contextmanager
def no_ytdlp():
    """Pretend the package is not installed."""
    saved = sys.modules.get("yt_dlp")
    sys.modules["yt_dlp"] = None          # `from None import X` raises ImportError
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("yt_dlp", None)
        else:
            sys.modules["yt_dlp"] = saved


class _Clock:
    """A hand-wound clock: sleeping moves it forward, nothing waits."""

    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, secs):
        self.slept.append(secs)
        self.t += secs


# ── which links are ours ────────────────────────────────────────────────────
@case("YouTube links in every shape the console will see are recognised")
def _():
    for url in ["https://www.youtube.com/watch?v=aqz-KE-bpKQ",
                "http://youtube.com/watch?v=aqz-KE-bpKQ",
                "https://m.youtube.com/watch?v=aqz-KE-bpKQ&list=PL1&index=3",
                "https://music.youtube.com/watch?v=aqz-KE-bpKQ",
                "https://youtu.be/aqz-KE-bpKQ?t=42",
                "https://www.youtube.com/shorts/abc12345678",
                "https://www.youtube.com/live/abc12345678",
                "https://www.youtube-nocookie.com/embed/abc12345678",
                "  https://www.youtube.com/watch?v=x  "]:
        assert is_youtube_url(url), url


@case("every other source kind is left to RtspCapture")
def _():
    # A false positive here would hand a working CCTV camera to yt-dlp.
    for url in ["rtsp://admin:pw@10.0.0.5:554/Streaming/Channels/102",
                "http://192.168.1.50:8080/video", "0", "1", "ws", "screen", "",
                r"C:\footage\clip.mp4", "https://example.com/video.mp4",
                "https://notyoutube.com/watch?v=x",
                "https://youtube.com.evil.example/watch?v=x",   # suffix trick
                "aqz-KE-bpKQ",                                  # bare id
                "ftp://youtube.com/x", "javascript:alert(1)"]:
        assert not is_youtube_url(url), url


@case("the video id is pulled out of each link shape, for display")
def _():
    for url, want in [("https://www.youtube.com/watch?v=aqz-KE-bpKQ", "aqz-KE-bpKQ"),
                      ("https://youtu.be/aqz-KE-bpKQ?t=42", "aqz-KE-bpKQ"),
                      ("https://www.youtube.com/shorts/abc12345678", "abc12345678"),
                      ("https://www.youtube.com/live/abc12345678", "abc12345678"),
                      ("https://www.youtube.com/embed/abc12345678", "abc12345678"),
                      ("https://m.youtube.com/watch?v=X1&list=PL&index=3", "X1"),
                      ("https://www.youtube.com/feed/subscriptions", "")]:
        assert video_id(url) == want, (url, video_id(url))


# ── expiry ──────────────────────────────────────────────────────────────────
@case("the expiry deadline is read from both URL shapes YouTube hands back")
def _():
    # Progressive MP4: a query parameter. HLS manifest: a path segment.
    assert expiry_of("https://rr7.googlevideo.com/videoplayback?expire=1789861986&ei=x") \
        == 1789861986.0
    assert expiry_of("https://manifest.googlevideo.com/api/manifest/hls_playlist"
                     "/expire/1789861988/ei/BB/ip/1.2.3.4/") == 1789861988.0


@case("a URL with no deadline is never treated as expired")
def _():
    for url in ["https://example.com/a.mp4", "", "not a url",
                "https://x/videoplayback?expire=soon"]:
        assert expiry_of(url) == 0.0, url


@case("a link is re-resolved before it expires, not after it has already failed")
def _():
    with fake_ytdlp(VOD_INFO):
        cap = YouTubeCapture(0, "https://youtu.be/x", refresh_margin_s=300)
        cap._resolve()
        first = cap._media_url
        assert first and cap.resolves == 1
        # Well inside its life: the cached URL is reused, no second lookup.
        cap._expires_at = _FakeYDL_now() + 3600
        assert cap._resolve_source() == first and cap.resolves == 1
        # Inside the refresh margin: resolved again before it can fail.
        cap._expires_at = _FakeYDL_now() + 120
        cap._resolve_source()
        assert cap.resolves == 2, "should have re-resolved inside the margin"


def _FakeYDL_now():
    import time
    return time.time()


@case("a failed open drops the cached link so the retry resolves afresh")
def _():
    # googlevideo links are time-limited AND bound to the IP that asked for
    # them. Retrying the same one just fails again, forever.
    with fake_ytdlp(VOD_INFO):
        cap = YouTubeCapture(0, "https://youtu.be/x")
        cap._resolve()
        assert cap._media_url
        orig = RtspCapture._open_capture
        RtspCapture._open_capture = lambda self: None        # open fails
        try:
            assert cap._open_capture() is None
        finally:
            RtspCapture._open_capture = orig
        assert cap._media_url == "", "a dead link must not be retried"


# ── format choice ───────────────────────────────────────────────────────────
@case("a recording opens the tallest H.264 stream within the cap, as one file")
def _():
    f = pick_format(VOD_INFO, max_height=720)
    assert f["format_id"] == "298", f["format_id"]      # 720p avc1, protocol https
    assert f["protocol"] == "https", "a seekable single file, so it can rewind"


@case("the height cap is honoured — frames above it are decoded and thrown away")
def _():
    assert pick_format(VOD_INFO, max_height=360)["height"] == 360
    assert pick_format(VOD_INFO, max_height=144)["height"] == 144
    assert pick_format(VOD_INFO, max_height=1080)["height"] == 1080


@case("H.264 wins over VP9 and AV1 at the same height")
def _():
    # Not cosmetic: the bundled FFmpeg decodes AV1 in software, slowly.
    assert pick_format(VOD_INFO, max_height=720)["vcodec"].startswith("avc1")
    only_720 = dict(VOD_INFO, formats=[f for f in VOD_INFO["formats"]
                                       if f.get("height") == 720])
    assert pick_format(only_720)["format_id"] == "298"


@case("a DASH segment manifest is never chosen — OpenCV cannot open one")
def _():
    dash_only = dict(VOD_INFO, formats=[
        _fmt("616", 720, "vp09.00.41.0", "http_dash_segments"),
        _fmt("134", 360, "avc1.4d401e", "https"),
    ])
    assert pick_format(dash_only)["format_id"] == "134", \
        "should drop to a smaller stream rather than pick DASH"


@case("audio-only rows and rows with no URL are skipped")
def _():
    assert pick_format(VOD_INFO)["vcodec"] != "none"
    broken = dict(VOD_INFO, formats=[
        dict(_fmt("298", 720, "avc1", "https"), url=None),
        _fmt("134", 360, "avc1.4d401e", "https"),
    ])
    assert pick_format(broken)["format_id"] == "134"


@case("a live stream takes HLS and ignores the unplayable https row")
def _():
    f = pick_format(LIVE_INFO, max_height=720)
    assert f["protocol"].startswith("m3u8"), f["protocol"]
    assert f["format_id"] == "95", f["format_id"]


@case("when everything is above the cap, the smallest is taken rather than nothing")
def _():
    big = dict(VOD_INFO, formats=[_fmt("299", 1080, "avc1", "https"),
                                  _fmt("401", 2160, "av01", "https")])
    assert pick_format(big, max_height=720)["height"] == 1080


@case("a video with nothing playable fails with a reason an operator can act on")
def _():
    for formats in ([], [{"format_id": "140", "vcodec": "none", "acodec": "mp4a",
                          "url": "https://x", "protocol": "https"}]):
        try:
            pick_format(dict(VOD_INFO, formats=formats))
        except RuntimeError as e:
            assert "age-restricted" in str(e) or "no formats" in str(e), e
        else:
            raise AssertionError("should have raised")


# ── what the capture decides ────────────────────────────────────────────────
@case("a recording is paced and replays from the start")
def _():
    with fake_ytdlp(VOD_INFO):
        cap = YouTubeCapture(0, "https://youtu.be/x")
        cap._resolve()
    assert cap.paced and cap.loop_at_end
    assert cap.is_live is False
    assert cap.kind == "youtube"
    assert cap.quality == "720p60", cap.quality
    assert cap.title == "A Clip"


@case("a live stream is paced too, but never rewound")
def _():
    # Pacing a live stream is not obvious: FFmpeg hands over a whole HLS
    # segment at once, and the decode throttle then spends its budget on the
    # first frames of each segment. Measured against a real 30 fps bridge cam,
    # unpaced delivered 0.3 fps; paced, 8.1.
    with fake_ytdlp(LIVE_INFO):
        cap = YouTubeCapture(0, "https://youtu.be/x")
        cap._resolve()
    assert cap.is_live is True
    assert cap.paced, "an HLS segment burst needs holding back too"
    assert not cap.loop_at_end, "a live stream has no beginning to go back to"


@case("loop_vod: false plays a recording once and then reconnects")
def _():
    with fake_ytdlp(VOD_INFO):
        cap = YouTubeCapture(0, "https://youtu.be/x", loop_vod=False)
        cap._resolve()
    assert cap.paced and not cap.loop_at_end


@case("a playlist or channel link plays its first video rather than failing")
def _():
    with fake_ytdlp({"_type": "playlist", "entries": [VOD_INFO]}):
        cap = YouTubeCapture(0, "https://youtube.com/playlist?list=PL1")
        cap._resolve()
    assert cap.title == "A Clip"


@case("the console gets the title, quality and time left on the link")
def _():
    with fake_ytdlp(VOD_INFO):
        cap = YouTubeCapture(7, "https://youtu.be/aqz-KE-bpKQ", name="Gate")
        cap._resolve()
    info = cap.info()
    assert info["kind"] == "youtube" and info["label"] == "Gate"
    y = info["youtube"]
    assert y["title"] == "A Clip" and y["quality"] == "720p60"
    assert y["is_live"] is False and y["duration_s"] == 635.0
    assert y["expires_in_s"] > 0 and y["resolves"] == 1
    assert y["resolve_error"] == ""
    # The RTSP block must survive: the console card reads it for every camera.
    assert "rtsp" in info and info["rtsp"]["url"].startswith("https://")


# ── failure modes an operator will actually hit ─────────────────────────────
@case("a missing yt-dlp disables just this camera, and says how to fix it")
def _():
    with no_ytdlp():
        cap = YouTubeCapture(0, "https://youtu.be/x")
        try:
            cap._resolve()
        except RuntimeError as e:
            assert "pip install yt-dlp" in str(e), e
        else:
            raise AssertionError("should have raised")
        assert "yt-dlp" in cap.resolve_error
        # _open_capture swallows it into last_error rather than killing the thread.
        assert cap._open_capture() is None
        assert "yt-dlp" in cap.last_error


@case("a yt-dlp error reaches the operator without its ANSI colour codes")
def _():
    err = Exception("\x1b[0;31mERROR:\x1b[0m [youtube] abc12345678: Video unavailable")
    with fake_ytdlp(error=err):
        cap = YouTubeCapture(0, "https://youtu.be/abc12345678")
        try:
            cap._resolve()
        except RuntimeError:
            pass
    assert cap.resolve_error == "Video unavailable", repr(cap.resolve_error)


@case("the Test button reports the video, or why it cannot be played")
def _():
    with fake_ytdlp(VOD_INFO):
        r = probe("https://youtu.be/x")
    assert r["ok"] and r["height"] == 720 and r["fps"] == 60
    assert r["title"] == "A Clip" and r["is_live"] is False
    assert "10:35" in r["note"], r["note"]              # 635 s
    with fake_ytdlp(error=Exception("ERROR: Private video")):
        r = probe("https://youtu.be/x")
    assert r["ok"] is False and "Private video" in r["error"]


@case("a live probe says LIVE instead of a duration")
def _():
    with fake_ytdlp(LIVE_INFO):
        r = probe("https://youtu.be/x")
    assert r["ok"] and r["is_live"] is True and "LIVE" in r["note"]


# ── pacing ──────────────────────────────────────────────────────────────────
@case("a paced source advances one frame per frame-period, not as fast as it can")
def _():
    # Without this a googlevideo URL demuxes at ~470 fps: the clip races past on
    # the dashboard, and the behaviour engine — which measures speed in body
    # heights per second of wall clock — reads every walk as a sprint.
    c = _Clock()
    p = _Pacer(30.0, clock=c.now, sleep=c.sleep)
    start = c.t
    for _ in range(90):
        p.wait()
    assert abs((c.t - start) - 3.0) < 0.05, c.t - start      # 90 frames @30 = 3 s
    assert p.resyncs == 0


@case("a source that paces itself is not slowed down at all")
def _():
    # RtspCapture must leave RTSP and webcams exactly as they were.
    cap = RtspCapture(0, "rtsp://10.0.0.5/stream")
    assert cap.paced is False and cap.loop_at_end is False
    assert RtspCapture(0, "0").paced is False
    assert RtspCapture(0, r"C:\clip.mp4").paced is True


@case("falling far behind drops the backlog instead of fast-forwarding to catch up")
def _():
    c = _Clock()
    p = _Pacer(30.0, clock=c.now, sleep=c.sleep)
    p.wait()
    c.t += 10.0                    # a slow decode, or a laptop waking up
    p.wait()
    assert p.resyncs == 1
    # Back on schedule from now: the next 30 frames take one second, they do
    # not race through ten seconds of video to make up the gap.
    t = c.t
    for _ in range(30):
        p.wait()
    assert abs((c.t - t) - 1.0) < 0.05, c.t - t


@case("an unreadable frame rate falls back to ordinary video, never to a stall")
def _():
    for bad in (0.0, None, -5.0, 1e9, float("nan")):
        p = _Pacer(bad)
        assert 1.0 <= p.fps <= _Pacer.MAX_FPS, (bad, p.fps)
    assert _Pacer(0.0).fps == 25.0


@case("rewinding resets the schedule so the replay is not judged late")
def _():
    c = _Clock()
    p = _Pacer(30.0, clock=c.now, sleep=c.sleep)
    for _ in range(30):
        p.wait()
    c.t += 5.0                     # the seek back to frame 0 took a moment
    p.reset()
    t = c.t
    for _ in range(30):
        p.wait()
    assert p.resyncs == 0, "a reset schedule must not look like lateness"
    assert abs((c.t - t) - 1.0) < 0.05


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
