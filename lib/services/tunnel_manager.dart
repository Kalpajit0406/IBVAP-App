import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';

import '../config/app_config.dart';

/// What the public tunnel forwards to.
enum TunnelTarget {
  /// The HTTPS camera-intake listener (:8443). The server's ListenerGuard only
  /// serves the phone page and its frame WebSocket there, so nothing else is
  /// reachable through the tunnel.
  phones('phones', 'Phone cameras only'),

  /// The HTTP console listener (:8090) — dashboard, API, video streams.
  console('console', 'Full console (dashboard + API)');

  const TunnelTarget(this.id, this.label);
  final String id;
  final String label;

  static TunnelTarget fromId(String id) =>
      values.firstWhere((t) => t.id == id, orElse: () => phones);
}

/// * [quick] — `cloudflared tunnel --url …`: free, no account, random
///   `*.trycloudflare.com` hostname that changes on every start.
/// * [named] — `cloudflared tunnel run --token …`: a tunnel created in the
///   Cloudflare dashboard with a public hostname on your own domain. The
///   hostname → service mapping lives in the dashboard.
enum TunnelKind { quick, named }

enum TunnelStatus {
  stopped,

  /// cloudflared launched, waiting for an edge connection.
  starting,

  /// Edge connection registered; checking the public address really answers.
  verifying,
  running,
  failed,
}

/// Runs `cloudflared` as a child process so a phone (or a remote viewer) can
/// reach this local product over a valid HTTPS address with no port
/// forwarding, firewall rule or certificate warning.
///
/// Start-up is defensive because every failure seen in the field looks the
/// same to an operator ("it timed out"):
///  * output is parsed line-by-line — cloudflared's banner and pre-check table
///    are several KB, so the "Registered tunnel connection" line routinely
///    straddles two pipe reads;
///  * the local listener is checked first, so a stopped backend is reported as
///    such instead of as a tunnel problem;
///  * each attempt gets a bounded wait, then the next attempt forces HTTP/2 on
///    TCP 443 (networks that block or throttle UDP/QUIC never register);
///  * trycloudflare rate limits (429 / 1015) are retried with back-off;
///  * after registration the public address is resolved through Cloudflare
///    DNS-over-HTTPS and fetched, so "TUNNEL UP" means a phone can load it;
///  * an unexpected cloudflared exit is restarted a few times.
class TunnelManager extends ChangeNotifier {
  TunnelManager({this.registerTimeout = const Duration(seconds: 30)});

  /// How long one cloudflared attempt may take to register an edge connection.
  final Duration registerTimeout;

  Process? _proc;
  TunnelStatus _status = TunnelStatus.stopped;
  String? _publicUrl;
  String? _error;
  String? _notice;
  final List<String> _log = [];
  Timer? _attemptTimer;
  String? _pendingUrl;
  bool _registered = false;
  bool _rateLimited = false;
  int _attempt = 0;
  int _session = 0;
  final List<DateTime> _restarts = [];
  IOSink? _logFile;

  static const _maxAttempts = 3;

  TunnelStatus get status => _status;
  bool get isRunning => _status == TunnelStatus.running;
  bool get isBusy =>
      _status == TunnelStatus.starting || _status == TunnelStatus.verifying;

  /// `https://…` with no trailing slash once the tunnel is up.
  String? get publicUrl => _publicUrl;
  String? get error => _error;

  /// Non-fatal information about the running tunnel (e.g. "DNS still
  /// propagating"), shown in amber.
  String? get notice => _notice;
  List<String> get log => List.unmodifiable(_log);

  TunnelKind get kind =>
      AppConfig.instance.tunnelToken.isNotEmpty ? TunnelKind.named : TunnelKind.quick;
  TunnelTarget get target => TunnelTarget.fromId(AppConfig.instance.tunnelTarget);

  /// Phone page URL for a slot, when phones can use the tunnel.
  String? phoneUrl(int camId) {
    final u = _publicUrl;
    if (u == null || !isRunning) return null;
    return '$u/cam/$camId';
  }

  static final _quickUrl = RegExp(r'https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com');

  /// Resolves the cloudflared executable: explicit setting, then a copy
  /// shipped next to the app, then Program Files, then PATH.
  static Future<String?> locateBinary() async {
    final configured = AppConfig.instance.cloudflaredPath;
    if (configured.isNotEmpty && await File(configured).exists()) return configured;

    final exeDir = File(Platform.resolvedExecutable).parent.path;
    final sep = Platform.pathSeparator;
    for (final c in [
      [exeDir, 'backend', 'tools', 'cloudflared.exe'].join(sep),
      [AppConfig.instance.backendDir, '..', 'tools', 'cloudflared.exe'].join(sep),
      '${Platform.environment['ProgramFiles'] ?? r'C:\Program Files'}${sep}cloudflared${sep}cloudflared.exe',
      '${Platform.environment['ProgramFiles(x86)'] ?? r'C:\Program Files (x86)'}${sep}cloudflared${sep}cloudflared.exe',
    ]) {
      if (await File(c).exists()) return c;
    }
    try {
      final r = await Process.run(Platform.isWindows ? 'where' : 'which', ['cloudflared']);
      if (r.exitCode == 0) {
        final first = r.stdout.toString().split(RegExp(r'\r?\n')).first.trim();
        if (first.isNotEmpty) return first;
      }
    } catch (_) {}
    return null;
  }

  int get _originPort => target == TunnelTarget.phones ? 8443 : _consolePort;

  int get _consolePort {
    final u = Uri.tryParse(AppConfig.instance.baseUrl);
    return (u != null && u.hasPort) ? u.port : 8090;
  }

  /// Attempt 1 lets cloudflared pick (QUIC, with its own pre-checks); later
  /// attempts force HTTP/2 over TCP 443, which passes almost any firewall.
  List<String> _args(int attempt) {
    final cfg = AppConfig.instance;
    final proto = attempt > 1 ? ['--protocol', 'http2'] : const <String>[];
    if (kind == TunnelKind.named) {
      return ['tunnel', '--no-autoupdate', ...proto, 'run', '--token', cfg.tunnelToken];
    }
    final origin = target == TunnelTarget.phones
        ? 'https://127.0.0.1:8443'
        : 'http://127.0.0.1:$_consolePort';
    return [
      'tunnel',
      '--no-autoupdate',
      ...proto,
      '--url', origin,
      // The :8443 listener uses the product's self-signed certificate.
      if (target == TunnelTarget.phones) '--no-tls-verify',
    ];
  }

  Future<bool> start() async {
    if (_status == TunnelStatus.running || isBusy) return true;
    _session++;
    _restarts.clear();
    _error = null;
    _notice = null;
    _publicUrl = null;
    _log.clear();
    _openLogFile();
    _status = TunnelStatus.starting;
    notifyListeners();

    final cfg = AppConfig.instance;
    if (kind == TunnelKind.named && cfg.tunnelHostname.isEmpty) {
      return _fail('Named tunnel needs its public hostname (e.g. stream.example.com).');
    }

    final bin = await locateBinary();
    if (bin == null) {
      return _fail('cloudflared is not installed. Install it with: '
          'winget install --id Cloudflare.cloudflared  (or set its path below).');
    }

    if (!await _originListening()) {
      return _fail(target == TunnelTarget.phones
          ? 'Nothing is listening on 127.0.0.1:8443 (phone intake). Turn the backend '
              'on in Phone Cameras or All Sources mode first.'
          : 'Nothing is listening on 127.0.0.1:$_consolePort. Turn the backend on first.');
    }

    return _launch(bin, 1, _session);
  }

  Future<bool> _originListening() async {
    try {
      final s = await Socket.connect('127.0.0.1', _originPort,
          timeout: const Duration(seconds: 2));
      s.destroy();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> _launch(String bin, int attempt, int session) async {
    _attempt = attempt;
    _pendingUrl = null;
    _registered = false;
    _rateLimited = false;
    final args = _args(attempt);
    _append('> [attempt $attempt/$_maxAttempts] ${_redacted(bin)} '
        '${args.map(_redacted).join(' ')}');
    notifyListeners();

    final Process p;
    try {
      p = await Process.start(bin, args);
    } catch (e) {
      return _fail('Could not start cloudflared: $e');
    }
    if (session != _session || _status == TunnelStatus.stopped) {
      _kill(p);
      return false;
    }
    _proc = p;
    for (final stream in [p.stdout, p.stderr]) {
      stream
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen((line) => _onLine(p, line), onError: (_) {});
    }
    p.exitCode.then((code) => _onExit(p, code, bin, session));

    _attemptTimer?.cancel();
    _attemptTimer = Timer(registerTimeout, () {
      if (!identical(_proc, p) || _status != TunnelStatus.starting) return;
      _append('! no edge connection after ${registerTimeout.inSeconds} s');
      _retryOrFail(p, bin, session,
          'Cloudflare did not accept a connection after $_maxAttempts attempts '
          '(QUIC and HTTP/2). Check that this PC has internet access and that '
          'outbound port 7844 or 443 is not blocked.');
    });
    return true;
  }

  void _retryOrFail(Process p, String bin, int session, String finalError) {
    _proc = null;
    _kill(p);
    if (_attempt >= _maxAttempts) {
      _fail(_rateLimited
          ? 'Cloudflare is rate-limiting free quick tunnels from this network. '
              'Wait a minute and try again, or use a Named tunnel.'
          : finalError);
      return;
    }
    final wait = Duration(seconds: _rateLimited ? 8 * _attempt : 1);
    _append('… retrying in ${wait.inSeconds} s over HTTP/2 (TCP 443)');
    notifyListeners();
    Future.delayed(wait, () {
      if (session != _session || _status != TunnelStatus.starting) return;
      _launch(bin, _attempt + 1, session);
    });
  }

  void _onLine(Process p, String line) {
    if (line.trim().isEmpty) return;
    _append(_redacted(line));
    if (!identical(_proc, p)) return;

    if (line.contains('Unauthorized') || line.contains('Invalid tunnel secret')) {
      _error = 'Cloudflare rejected the tunnel token.';
    }
    if (line.contains('429') || line.contains('1015') ||
        line.toLowerCase().contains('too many requests')) {
      _rateLimited = true;
    }

    if (_status == TunnelStatus.starting) {
      final m = _quickUrl.firstMatch(line);
      if (m != null) _pendingUrl = m.group(0);
      if (line.contains('Registered tunnel connection')) _registered = true;
      // cloudflared prints the quick-tunnel URL before the edge connection is
      // registered; handing out the URL then gives phones a Cloudflare 530.
      if (_registered) {
        String? url;
        if (kind == TunnelKind.quick) {
          url = _pendingUrl;
        } else {
          final host = AppConfig.instance.tunnelHostname
              .replaceFirst(RegExp(r'^https?://'), '')
              .replaceAll(RegExp(r'/+$'), '');
          url = 'https://$host';
        }
        if (url != null) _onRegistered(url);
      }
    }
    notifyListeners();
  }

  void _onRegistered(String url) {
    _attemptTimer?.cancel();
    _publicUrl = url;
    _status = TunnelStatus.verifying;
    _error = null;
    _append('✓ edge connection registered — verifying $url');
    _verify(url, _session);
  }

  /// Confirms the public address loads from the internet side. A brand-new
  /// trycloudflare hostname takes a few seconds to appear in DNS, and Windows
  /// caches the "not found" answer for minutes, so resolve through Cloudflare
  /// DNS-over-HTTPS and connect to that address directly (still validating the
  /// certificate against the real hostname).
  Future<void> _verify(String url, int session) async {
    final host = Uri.parse(url).host;
    final path = target == TunnelTarget.phones ? '/cam/0' : '/status';
    final deadline = DateTime.now().add(const Duration(seconds: 60));
    String? lastProblem;
    while (DateTime.now().isBefore(deadline)) {
      if (session != _session || _status != TunnelStatus.verifying) return;
      final ip = await _resolveDoh(host);
      if (ip == null) {
        lastProblem = 'public DNS has not published $host yet';
      } else {
        final code = await _probe(ip, host, path);
        if (session != _session || _status != TunnelStatus.verifying) return;
        if (code != null && code < 500) {
          _append('✓ public check: GET $path → $code via $ip');
          _up(url, null);
          _flushLocalDns();
          return;
        }
        lastProblem = code == null
            ? 'could not reach $ip:443'
            : code == 502
                ? 'Cloudflare reached this PC but the backend did not answer (502)'
                : 'public address answered HTTP $code';
      }
      _append('… $lastProblem — retrying');
      notifyListeners();
      await Future.delayed(const Duration(seconds: 3));
    }
    if (session != _session || _status != TunnelStatus.verifying) return;
    // The tunnel itself is registered; only the outside check is inconclusive
    // (e.g. DoH blocked on this network). Hand out the URL, but say so.
    _up(url, 'Tunnel is registered but the public check did not pass '
        '($lastProblem). A phone may need a minute before the address loads.');
    _flushLocalDns();
  }

  static Future<String?> _resolveDoh(String host) async {
    for (final base in const [
      'https://1.1.1.1/dns-query',
      'https://cloudflare-dns.com/dns-query',
      'https://dns.google/resolve',
    ]) {
      final client = HttpClient()..connectionTimeout = const Duration(seconds: 4);
      try {
        final req = await client
            .getUrl(Uri.parse('$base?name=$host&type=A'))
            .timeout(const Duration(seconds: 5));
        req.headers.set(HttpHeaders.acceptHeader, 'application/dns-json');
        final res = await req.close().timeout(const Duration(seconds: 5));
        final body = await res.transform(utf8.decoder).join();
        final j = jsonDecode(body);
        if (j is Map && j['Answer'] is List) {
          for (final a in j['Answer'] as List) {
            if (a is Map && a['type'] == 1 && a['data'] is String) return a['data'] as String;
          }
        }
        if (j is Map && j['Status'] is int) return null; // authoritative: not there yet
      } catch (_) {
        // try the next resolver
      } finally {
        client.close(force: true);
      }
    }
    return null;
  }

  /// Raw HTTPS GET to [ip] with SNI + certificate validation for [host];
  /// returns the status code or null on a connection failure.
  static Future<int?> _probe(String ip, String host, String path) async {
    SecureSocket? tls;
    try {
      final raw = await Socket.connect(ip, 443, timeout: const Duration(seconds: 6));
      tls = await SecureSocket.secure(raw, host: host)
          .timeout(const Duration(seconds: 8));
      tls.write('GET $path HTTP/1.1\r\nHost: $host\r\n'
          'User-Agent: IBVAP-tunnel-check\r\nConnection: close\r\n\r\n');
      await tls.flush();
      final head = await tls
          .cast<List<int>>()
          .transform(latin1.decoder)
          .transform(const LineSplitter())
          .first
          .timeout(const Duration(seconds: 15));
      final m = RegExp(r'^HTTP/\d(?:\.\d)?\s+(\d{3})').firstMatch(head);
      return m == null ? null : int.parse(m.group(1)!);
    } catch (_) {
      return null;
    } finally {
      tls?.destroy();
    }
  }

  /// Drops Windows' cached "host not found" so the copied link also opens in
  /// a browser on this PC. Needs no elevation.
  static void _flushLocalDns() {
    if (!Platform.isWindows) return;
    Process.run('ipconfig', ['/flushdns']).ignore();
  }

  void _up(String url, String? notice) {
    _attemptTimer?.cancel();
    _publicUrl = url;
    _status = TunnelStatus.running;
    _error = null;
    _notice = notice;
    if (notice != null) _append('! $notice');
    notifyListeners();
  }

  void _onExit(Process p, int code, String bin, int session) {
    _append('cloudflared exited (code $code)');
    if (!identical(_proc, p) || session != _session) return;
    _proc = null;
    switch (_status) {
      case TunnelStatus.starting:
        _retryOrFail(p, bin, session,
            _error ?? 'cloudflared exited (code $code) — see the tunnel log.');
      case TunnelStatus.verifying:
      case TunnelStatus.running:
        final now = DateTime.now();
        _restarts.removeWhere((t) => now.difference(t) > const Duration(minutes: 10));
        if (_restarts.length >= 3) {
          _publicUrl = null;
          _fail('cloudflared keeps exiting (code $code) — see the tunnel log.');
          return;
        }
        _restarts.add(now);
        _publicUrl = null;
        _status = TunnelStatus.starting;
        _notice = kind == TunnelKind.quick
            ? 'The tunnel dropped and is reconnecting — quick tunnels get a NEW '
                'address, so phones must re-scan.'
            : 'The tunnel dropped and is reconnecting.';
        notifyListeners();
        Future.delayed(const Duration(seconds: 2), () {
          if (session == _session && _status == TunnelStatus.starting) {
            _launch(bin, 1, session);
          }
        });
      case TunnelStatus.stopped:
      case TunnelStatus.failed:
        break;
    }
  }

  Future<void> stop({bool keepError = false}) async {
    _session++;
    _attemptTimer?.cancel();
    final p = _proc;
    _proc = null;
    _publicUrl = null;
    _notice = null;
    _status = keepError ? TunnelStatus.failed : TunnelStatus.stopped;
    if (!keepError) _error = null;
    notifyListeners();
    if (p != null) await _kill(p);
  }

  static Future<void> _kill(Process p) async {
    p.kill();
    if (Platform.isWindows) {
      try {
        await Process.run('taskkill', ['/F', '/T', '/PID', '${p.pid}']);
      } catch (_) {}
    }
  }

  bool _fail(String msg) {
    _attemptTimer?.cancel();
    _error = msg;
    _status = TunnelStatus.failed;
    final p = _proc;
    _proc = null;
    if (p != null) _kill(p);
    _append('! $msg');
    notifyListeners();
    return false;
  }

  /// The log is also written to `%APPDATA%\ibvap_app\tunnel.log` so a failed
  /// start can be diagnosed after the fact.
  void _openLogFile() {
    try {
      _logFile?.close();
      final root = Platform.environment['APPDATA'] ?? Directory.systemTemp.path;
      final f = File('$root${Platform.pathSeparator}ibvap_app'
          '${Platform.pathSeparator}tunnel.log');
      f.parent.createSync(recursive: true);
      _logFile = f.openWrite();
    } catch (_) {
      _logFile = null;
    }
  }

  void _append(String line) {
    _log.add(line);
    if (_log.length > 400) _log.removeRange(0, _log.length - 400);
    try {
      _logFile?.writeln('${DateTime.now().toIso8601String()} $line');
    } catch (_) {}
  }

  /// Never let the tunnel token land in the visible log.
  static String _redacted(String s) {
    final tok = AppConfig.instance.tunnelToken;
    return tok.length > 8 ? s.replaceAll(tok, '<token>') : s;
  }

  bool _disposed = false;

  @override
  void notifyListeners() {
    if (!_disposed) super.notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    _session++;
    _attemptTimer?.cancel();
    final p = _proc;
    _proc = null;
    if (p != null) {
      p.kill();
      if (Platform.isWindows) {
        Process.run('taskkill', ['/F', '/T', '/PID', '${p.pid}']).ignore();
      }
    }
    _logFile?.close();
    super.dispose();
  }
}
