import 'dart:convert';
import 'dart:io';

/// Runtime configuration for the IBVAP client.
///
/// The IBVAP server runs two listeners:
///   * HTTP  :8090 — dashboard + API (`/status`, `/api/*`, `/snap/*`, `/stream`)
///   * HTTPS :8443 — phone camera intake (not used by this client)
/// so the default points at the HTTP one on localhost.
///
/// Persisted as a small JSON file under the OS app-data dir — no plugins, so
/// the Windows build stays dependency-light.
class AppConfig {
  AppConfig._() {
    _backendDir = _defaultBackendDir();
    _pythonPath = _defaultPythonPath();
  }
  static final AppConfig instance = AppConfig._();

  static const defaultBaseUrl = 'http://127.0.0.1:8090';
  static const _minPoll = 300;
  static const _maxPoll = 10000;

  String _baseUrl = defaultBaseUrl;
  int _pollIntervalMs = 1000;
  late String _backendDir;
  late String _pythonPath;
  String _launchMode = 'all';
  String _apiToken = '';
  bool _alarmSound = true;
  String _cloudflaredPath = '';
  String _tunnelToken = '';
  String _tunnelHostname = '';
  String _tunnelTarget = 'phones';

  String? _localTokenCache;
  DateTime? _localTokenReadAt;

  /// The folder the running exe lives in — e.g. `...\Release\` once built,
  /// or the `Debug\` runner folder under `flutter run`.
  static String get _exeDir => File(Platform.resolvedExecutable).parent.path;

  /// The product ships its own Python backend + trained models right next to
  /// the executable (`backend/app` — a vendored copy of the IBVAP server;
  /// `backend/runtime` — a self-contained Python venv with torch/CUDA/
  /// ultralytics/etc already installed), so the app needs nothing external
  /// installed or present elsewhere on the machine. If that bundle isn't
  /// found (e.g. a dev build run straight from `build\windows\...\Debug`
  /// without the packaging step), fall back to a dev checkout + system
  /// Python so `flutter run` keeps working during development.
  static String _defaultBackendDir() {
    final root = _productRoot();
    if (root != null) return '$root${Platform.pathSeparator}app';
    return _legacyDevBackendDir;
  }

  static String _defaultPythonPath() {
    final root = _productRoot();
    if (root != null) {
      final py = '$root${Platform.pathSeparator}runtime${Platform.pathSeparator}Scripts'
          '${Platform.pathSeparator}python.exe';
      if (File(py).existsSync()) return py;
    }
    return 'python';
  }

  /// Values an older dev build used to persist. They pointed a build at a
  /// separate source checkout instead of this product's own backend.
  static const _legacyDevBackendDir = r'E:\IBVAP';

  /// The product's `backend` folder: next to the exe once packaged, or — for
  /// a `flutter run` Debug/Profile build under `build\windows\x64\runner\…` —
  /// the project's own `backend` folder, so dev builds run the very same
  /// bundled server + runtime as the shipped product.
  static String? _productRoot() {
    final sep = Platform.pathSeparator;
    var dir = Directory(_exeDir);
    for (var i = 0; i < 7; i++) {
      if (File('${dir.path}${sep}backend${sep}app${sep}server.py').existsSync()) {
        return '${dir.path}${sep}backend';
      }
      final parent = dir.parent;
      if (parent.path == dir.path) break;
      dir = parent;
    }
    return null;
  }

  /// True when this launch found the bundled backend next to the exe (i.e.
  /// the app is running as the packaged standalone product, not a dev build
  /// pointed at an external checkout).
  bool get isBundled => _productRoot() != null;

  /// What [backendDir] / [pythonPath] would be right now if reset to the
  /// bundled defaults — for a Settings "Reset to bundled" action, independent
  /// of whatever override is currently loaded.
  static String get defaultBackendDirForDisplay => _defaultBackendDir();
  static String get defaultPythonPathForDisplay => _defaultPythonPath();

  /// Normalised base URL with no trailing slash, e.g. `http://10.0.0.4:8090`.
  String get baseUrl => _baseUrl;

  /// `/status` poll cadence in milliseconds (clamped 300–10000).
  int get pollIntervalMs => _pollIntervalMs;

  /// Directory path where the IBVAP Python backend is located (e.g. `E:\IBVAP`).
  String get backendDir => _backendDir;

  /// Path to Python executable (e.g. `python` or `python.exe` or full venv path).
  String get pythonPath => _pythonPath;

  /// Default launch mode ('all', 'screen', 'phone', 'cctv', 'demo').
  String get launchMode => _launchMode;

  /// Token typed in Settings — needed for write actions on a remote server.
  String get apiToken => _apiToken;

  bool get alarmSound => _alarmSound;

  /// Explicit path to cloudflared.exe; empty = auto-detect.
  String get cloudflaredPath => _cloudflaredPath;

  /// Cloudflare dashboard tunnel token. Empty = free quick tunnel.
  String get tunnelToken => _tunnelToken;

  /// Public hostname routed to a named tunnel, e.g. stream.example.com.
  String get tunnelHostname => _tunnelHostname;

  /// 'phones' (camera intake only) or 'console' (dashboard + API).
  String get tunnelTarget => _tunnelTarget;

  bool get isLoopbackServer {
    final host = Uri.tryParse(_baseUrl)?.host ?? '';
    return host == '127.0.0.1' || host == 'localhost' || host == '::1' ||
        host == '[::1]';
  }

  /// The token to send: the Settings value, else — only when talking to this
  /// machine — the local backend's own `data/api_token`. The local token is
  /// never sent to a remote host.
  String? get effectiveApiToken {
    if (_apiToken.isNotEmpty) return _apiToken;
    if (!isLoopbackServer) return null;
    final now = DateTime.now();
    if (_localTokenReadAt == null ||
        now.difference(_localTokenReadAt!) > const Duration(seconds: 30)) {
      _localTokenReadAt = now;
      try {
        final f = File('$_backendDir${Platform.pathSeparator}data'
            '${Platform.pathSeparator}api_token');
        _localTokenCache = f.existsSync() ? f.readAsStringSync().trim() : null;
      } catch (_) {
        _localTokenCache = null;
      }
    }
    return _localTokenCache;
  }

  Uri endpoint(String path) {
    final p = path.startsWith('/') ? path : '/$path';
    return Uri.parse('$_baseUrl$p');
  }

  /// Null when [raw] normalises to a usable http(s) base URL, else the reason.
  static String? validateBaseUrl(String raw) {
    if (raw.trim().isEmpty) return 'Enter a server address.';
    final scheme = RegExp(r'^([a-zA-Z][a-zA-Z0-9+.-]*)://').firstMatch(raw.trim());
    if (scheme != null && !{'http', 'https'}.contains(scheme.group(1)!.toLowerCase())) {
      return 'Use http:// or https://.';
    }
    final u = Uri.tryParse(_normalise(raw));
    if (u == null || u.host.isEmpty) return 'Not a valid address.';
    if (u.scheme != 'http' && u.scheme != 'https') return 'Use http:// or https://.';
    if (u.hasPort && (u.port <= 0 || u.port > 65535)) return 'Port out of range.';
    return null;
  }

  File get _file {
    final root = Platform.environment['APPDATA'] ??
        Platform.environment['HOME'] ??
        Directory.current.path;
    return File('$root${Platform.pathSeparator}ibvap_app'
        '${Platform.pathSeparator}config.json');
  }

  Future<void> load() async {
    try {
      final f = _file;
      if (await f.exists()) {
        final j = jsonDecode(await f.readAsString());
        if (j is Map) {
          final saved = j['server_base_url'] as String?;
          if (saved != null && saved.trim().isNotEmpty) {
            _baseUrl = _normalise(saved);
          }
          final poll = j['poll_interval_ms'];
          if (poll is num) _pollIntervalMs = _clampPoll(poll.toInt());
          // Empty = use the product's own backend. A stale dev-checkout path
          // saved by an older build is ignored whenever the bundled backend
          // exists, so the app never silently runs a different server.
          final bundledRoot = _productRoot();
          final bDir = (j['backend_dir'] as String?)?.trim() ?? '';
          final legacy = bundledRoot != null &&
              bDir.toLowerCase() == _legacyDevBackendDir.toLowerCase();
          if (bDir.isNotEmpty && !legacy) _backendDir = bDir;
          final py = (j['python_path'] as String?)?.trim() ?? '';
          if (py.isNotEmpty && !(legacy && py == 'python')) _pythonPath = py;
          final lm = j['launch_mode'] as String?;
          if (lm != null && lm.trim().isNotEmpty) {
            _launchMode = lm.trim();
          }
          final tok = j['api_token'];
          if (tok is String) _apiToken = tok.trim();
          final snd = j['alarm_sound'];
          if (snd is bool) _alarmSound = snd;
          String str(String k) => (j[k] is String) ? (j[k] as String).trim() : '';
          _cloudflaredPath = str('cloudflared_path');
          _tunnelToken = str('tunnel_token');
          _tunnelHostname = str('tunnel_hostname');
          if (str('tunnel_target').isNotEmpty) _tunnelTarget = str('tunnel_target');
        }
      }
    } catch (_) {
      // fall back to defaults — a missing / corrupt config is not fatal
    }
  }

  Future<void> setBaseUrl(String value) async {
    _baseUrl = _normalise(value);
    await _persist();
  }

  Future<void> setPollIntervalMs(int value) async {
    _pollIntervalMs = _clampPoll(value);
    await _persist();
  }

  Future<void> setBackendDir(String value) async {
    _backendDir = value.trim();
    _localTokenReadAt = null;
    await _persist();
  }

  Future<void> setPythonPath(String value) async {
    _pythonPath = value.trim();
    await _persist();
  }

  Future<void> setLaunchMode(String value) async {
    _launchMode = value.trim();
    await _persist();
  }

  Future<void> setApiToken(String value) async {
    _apiToken = value.trim();
    await _persist();
  }

  Future<void> setTunnel({
    String? cloudflaredPath,
    String? token,
    String? hostname,
    String? target,
  }) async {
    if (cloudflaredPath != null) _cloudflaredPath = cloudflaredPath.trim();
    if (token != null) _tunnelToken = token.trim();
    if (hostname != null) _tunnelHostname = hostname.trim();
    if (target != null) _tunnelTarget = target;
    await _persist();
  }

  Future<void> setAlarmSound(bool value) async {
    _alarmSound = value;
    await _persist();
  }

  Future<void> _persist() async {
    // Tests exercise setters too; they must never write the operator's real
    // config (a test run once pinned backend_dir/python_path to dev values,
    // which the packaged app would then have loaded).
    if (Platform.environment['FLUTTER_TEST'] == 'true') return;
    try {
      final f = _file;
      await f.parent.create(recursive: true);
      // Write-then-rename so a crash mid-save can't leave a truncated config
      // that silently resets every setting to defaults on the next launch.
      final tmp = File('${f.path}.tmp');
      await tmp.writeAsString(jsonEncode({
        'server_base_url': _baseUrl,
        'poll_interval_ms': _pollIntervalMs,
        // Only a genuine override is stored, so a moved install keeps finding
        // its bundled backend.
        'backend_dir': _backendDir == _defaultBackendDir() ? '' : _backendDir,
        'python_path': _pythonPath == _defaultPythonPath() ? '' : _pythonPath,
        'launch_mode': _launchMode,
        'api_token': _apiToken,
        'alarm_sound': _alarmSound,
        'cloudflared_path': _cloudflaredPath,
        'tunnel_token': _tunnelToken,
        'tunnel_hostname': _tunnelHostname,
        'tunnel_target': _tunnelTarget,
      }), flush: true);
      await tmp.rename(f.path);
    } catch (_) {
      // keep the in-memory value even if persistence fails
    }
  }

  static int _clampPoll(int v) =>
      v < _minPoll ? _minPoll : (v > _maxPoll ? _maxPoll : v);

  static String _normalise(String raw) {
    var v = raw.trim();
    if (!v.startsWith('http://') && !v.startsWith('https://')) {
      v = 'http://$v';
    }
    while (v.endsWith('/')) {
      v = v.substring(0, v.length - 1);
    }
    return v;
  }
}
