import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../theme.dart';

/// Renders a `multipart/x-mixed-replace` MJPEG stream (the IBVAP `/stream`
/// mosaic, or `/stream/<cam>`). Flutter's `Image.network` cannot do multipart,
/// so this reads the response as a byte stream and slices out each JPEG by its
/// SOI (FFD8) / EOI (FFD9) markers.
///
/// A connection that stays open but stops delivering frames is shown as
/// SIGNAL LOST over the last frame instead of silently freezing on it — a
/// frozen picture that looks live is the most dangerous failure a monitoring
/// console can have.
class MjpegView extends StatefulWidget {
  const MjpegView({
    super.key,
    required this.url,
    this.fit = BoxFit.contain,
    this.stallAfter = const Duration(seconds: 3),
    this.connectTimeout = const Duration(seconds: 6),
  });

  final Uri url;
  final BoxFit fit;
  final Duration stallAfter;
  final Duration connectTimeout;

  @override
  State<MjpegView> createState() => _MjpegViewState();
}

class _MjpegViewState extends State<MjpegView> {
  http.Client? _client;
  StreamSubscription<List<int>>? _sub;
  Timer? _retry;
  Timer? _watchdog;
  final BytesBuilder _buf = BytesBuilder(copy: false);
  Uint8List? _frame;
  DateTime? _frameAt;
  Object? _error;
  int _session = 0; // bumped on every (re)connect; stale callbacks are ignored
  int _backoffMs = 1000;

  static const _maxBuffer = 4 << 20;

  @override
  void initState() {
    super.initState();
    _connect();
    _watchdog = Timer.periodic(const Duration(milliseconds: 500), (_) {
      if (!mounted) return;
      final stalled = _isStalled;
      if (stalled || stalled != _lastStalled) {
        _lastStalled = stalled;
        setState(() {}); // also ticks the "last frame Ns ago" counter
      }
      // Open socket, no frames for a long time: force a reconnect.
      final since = _lastActivity;
      if (since != null &&
          DateTime.now().difference(since) > widget.stallAfter * 3 &&
          _retry == null) {
        _fail(TimeoutException('no frames'));
      }
    });
  }

  bool _lastStalled = false;
  DateTime? _connectAt;

  /// Later of the last frame and the last connect attempt — the stall clock
  /// restarts on each reconnect so the watchdog can't fire in a tight loop.
  DateTime? get _lastActivity {
    final f = _frameAt, c = _connectAt;
    if (f == null) return c;
    if (c == null) return f;
    return f.isAfter(c) ? f : c;
  }

  bool get _isStalled =>
      _frame != null &&
      _frameAt != null &&
      DateTime.now().difference(_frameAt!) > widget.stallAfter;

  @override
  void didUpdateWidget(covariant MjpegView old) {
    super.didUpdateWidget(old);
    if (old.url != widget.url) {
      _retry?.cancel();
      _retry = null;
      _teardown();
      _frame = null;
      _frameAt = null;
      _error = null;
      _backoffMs = 1000;
      _connect();
    }
  }

  Future<void> _connect() async {
    final session = ++_session;
    _error = null;
    _retry = null;
    _connectAt = DateTime.now();
    final client = http.Client();
    _client = client;
    try {
      final req = http.Request('GET', widget.url)
        ..headers['Accept'] = 'multipart/x-mixed-replace';
      final res = await client.send(req).timeout(widget.connectTimeout);
      if (session != _session) {
        client.close();
        return;
      }
      if (res.statusCode >= 400) {
        throw http.ClientException('HTTP ${res.statusCode}', widget.url);
      }
      _sub = res.stream.listen(
        (chunk) => _onData(session, chunk),
        onError: (Object e) => _failIf(session, e),
        onDone: () =>
            _failIf(session, http.ClientException('stream closed', widget.url)),
        cancelOnError: true,
      );
    } catch (e) {
      _failIf(session, e);
    }
  }

  void _onData(int session, List<int> chunk) {
    if (session != _session) return;
    _buf.add(chunk);
    if (_buf.length > _maxBuffer) {
      _buf.clear(); // runaway guard: corrupt / non-JPEG stream
      return;
    }
    // Only scan when a whole frame could be present; slice out the newest
    // complete JPEG and drop anything older (no point decoding a backlog).
    final bytes = _buf.toBytes();
    Uint8List? latest;
    var consumed = 0;
    var from = 0;
    while (true) {
      final start = _indexOf(bytes, 0xD8, from);
      if (start < 0) break;
      final end = _indexOf(bytes, 0xD9, start + 2);
      if (end < 0) break;
      latest = Uint8List.sublistView(bytes, start, end + 2);
      consumed = end + 2;
      from = consumed;
    }
    if (latest == null) return;
    final rest = Uint8List.sublistView(bytes, consumed);
    _buf.clear();
    if (rest.isNotEmpty) _buf.add(Uint8List.fromList(rest));
    if (!mounted) return;
    _backoffMs = 1000;
    setState(() {
      _frame = Uint8List.fromList(latest!);
      _frameAt = DateTime.now();
      _error = null;
    });
  }

  void _failIf(int session, Object e) {
    if (session == _session) _fail(e);
  }

  void _fail(Object e) {
    _teardown();
    if (!mounted) return;
    setState(() => _error = e);
    _retry?.cancel();
    _retry = Timer(Duration(milliseconds: _backoffMs), () {
      _retry = null;
      if (mounted) _connect();
    });
    _backoffMs = (_backoffMs * 2).clamp(1000, 8000);
  }

  void _teardown() {
    _session++;
    _sub?.cancel();
    _sub = null;
    _client?.close();
    _client = null;
    _buf.clear();
  }

  @override
  void dispose() {
    _retry?.cancel();
    _watchdog?.cancel();
    _teardown();
    super.dispose();
  }

  /// Index of a 0xFF,[second] marker pair at or after [from], else -1.
  static int _indexOf(Uint8List hay, int second, int from) {
    final end = hay.length - 1;
    for (var i = from; i < end; i++) {
      if (hay[i] == 0xFF && hay[i + 1] == second) return i;
    }
    return -1;
  }

  @override
  Widget build(BuildContext context) {
    final frame = _frame;
    if (frame == null) {
      return Container(
        color: Colors.black,
        alignment: Alignment.center,
        child: _error != null
            ? const _SignalState(
                icon: Icons.videocam_off_outlined,
                title: 'NO SIGNAL',
                note: 'Stream unavailable — retrying')
            : const _SignalState(
                icon: Icons.sensors, title: 'ACQUIRING', note: 'Connecting to stream'),
      );
    }
    final stalled = _isStalled || _error != null;
    return Stack(
      fit: StackFit.expand,
      children: [
        Image.memory(
          frame,
          fit: widget.fit,
          gaplessPlayback: true,
          filterQuality: FilterQuality.low,
        ),
        if (stalled)
          IgnorePointer(
            child: Container(
              color: const Color(0x99000000),
              alignment: Alignment.center,
              child: _SignalState(
                icon: Icons.signal_wifi_connected_no_internet_4_outlined,
                title: 'SIGNAL LOST',
                note: _frameAt == null
                    ? 'Reconnecting'
                    : 'Last frame ${_age(_frameAt!)} ago — picture is NOT live',
                color: IbvapColors.red,
              ),
            ),
          ),
      ],
    );
  }

  static String _age(DateTime t) {
    final s = DateTime.now().difference(t).inSeconds;
    return s < 60 ? '${s}s' : '${s ~/ 60}m ${s % 60}s';
  }
}

class _SignalState extends StatelessWidget {
  const _SignalState({
    required this.icon,
    required this.title,
    required this.note,
    this.color = IbvapColors.muted,
  });

  final IconData icon;
  final String title;
  final String note;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: color, size: 30),
        const SizedBox(height: 8),
        Text(title, style: IbvapText.label(size: 12, color: color)),
        const SizedBox(height: 4),
        Text(note, style: const TextStyle(color: IbvapColors.muted, fontSize: 11)),
      ],
    );
  }
}
