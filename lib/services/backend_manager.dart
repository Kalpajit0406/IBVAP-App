import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../config/app_config.dart';

enum BackendMode {
  all(
    id: 'all',
    label: 'All Sources (config.yaml)',
    description: 'Opens configured CCTV RTSP streams and mobile phone slots.',
    script: 'server.py',
    args: ['--mode', 'all'],
  ),
  screen(
    id: 'screen',
    label: 'Screen Capture (CAM-00)',
    description: 'Grabs the primary screen and runs full detection + pose.',
    script: 'server.py',
    args: ['--mode', 'screen'],
  ),
  phone(
    id: 'phone',
    label: 'Phone Cameras (WebSockets)',
    description: 'Opens mobile intake slots via browser WebSockets.',
    script: 'server.py',
    args: ['--mode', 'phone'],
  ),
  cctv(
    id: 'cctv',
    label: 'Real CCTV / RTSP',
    description: 'Pulls RTSP/NVR camera feeds defined in config.yaml.',
    script: 'server.py',
    args: ['--mode', 'cctv'],
  ),
  demo(
    id: 'demo',
    label: '4-Camera Demo Replay',
    description: 'Replays 4 synchronized 720p/24fps video feeds on GPU.',
    script: 'run_demo.py',
    args: ['--cams', '4'],
  );

  const BackendMode({
    required this.id,
    required this.label,
    required this.description,
    required this.script,
    required this.args,
  });

  final String id;
  final String label;
  final String description;
  final String script;
  final List<String> args;

  static BackendMode fromId(String id) {
    return BackendMode.values.firstWhere(
      (m) => m.id.toLowerCase() == id.toLowerCase(),
      orElse: () => BackendMode.all,
    );
  }
}

enum BackendStatus { stopped, starting, running, stopping, failed }

/// Manages the local Python IBVAP backend process (start, stop, restart,
/// logs streaming, and readiness health probing) directly from the Flutter app.
class BackendManager extends ChangeNotifier {
  BackendManager() {
    _mode = BackendMode.fromId(AppConfig.instance.launchMode);
  }

  Process? _process;
  BackendStatus _status = BackendStatus.stopped;
  BackendMode _mode = BackendMode.all;
  String? _lastError;
  int? _pid;
  DateTime? _startTime;

  final List<String> _logs = [];
  static const int _maxLogLines = 1500;

  BackendStatus get status => _status;
  BackendMode get mode => _mode;
  String? get lastError => _lastError;
  int? get pid => _pid;
  DateTime? get startTime => _startTime;
  List<String> get logs => List.unmodifiable(_logs);
  bool get isRunning => _status == BackendStatus.running;
  bool get isStarting => _status == BackendStatus.starting;
  bool get isStopped => _status == BackendStatus.stopped;
  bool get isFailed => _status == BackendStatus.failed;

  String get logsText => _logs.join('\n');

  Duration? get uptime =>
      _startTime != null ? DateTime.now().difference(_startTime!) : null;

  void setMode(BackendMode newMode) {
    if (_mode == newMode) return;
    _mode = newMode;
    AppConfig.instance.setLaunchMode(newMode.id);
    notifyListeners();
  }

  /// Fires on every appended / cleared log line. Separate from this notifier
  /// (lifecycle only) so the console shell doesn't rebuild at logging rate.
  final ValueNotifier<int> logRevision = ValueNotifier<int>(0);

  void clearLogs() {
    _logs.clear();
    if (!_disposed) logRevision.value++;
  }

  void _appendLog(String text) {
    final timestamp = DateTime.now().toIso8601String().substring(11, 19);
    for (final line in text.split(RegExp(r'\r?\n'))) {
      if (line.isEmpty) continue;
      _logs.add('[$timestamp] $line');
      if (_logs.length > _maxLogLines) {
        _logs.removeRange(0, _logs.length - _maxLogLines);
      }
    }
    if (!_disposed) logRevision.value++;
  }

  /// Start the IBVAP Python backend process.
  Future<bool> start({BackendMode? mode}) async {
    if (_status == BackendStatus.running || _status == BackendStatus.starting) {
      return true;
    }

    if (mode != null) {
      setMode(mode);
    }

    final cfg = AppConfig.instance;
    final backendDir = Directory(cfg.backendDir);
    final pythonExe = cfg.pythonPath;

    _status = BackendStatus.starting;
    _lastError = null;
    notifyListeners();

    if (await _serverAnswers()) {
      // Launching now would spawn a process that dies on "address in use"
      // while the readiness probe sees the *existing* server and reports
      // success — the console would then show a PID for a dead process.
      const err = 'A backend is already running on this address. '
          'Use it as-is, or stop it before starting a new one.';
      _appendLog('⚠️ $err (${cfg.baseUrl})');
      _status = BackendStatus.stopped;
      _lastError = err;
      notifyListeners();
      return false;
    }

    _appendLog('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    _appendLog('▶ Launching IBVAP Backend: ${_mode.label}');
    _appendLog('  Directory: ${backendDir.path}');
    _appendLog('  Python:    $pythonExe');
    _appendLog('  Command:   $pythonExe ${_mode.script} ${_mode.args.join(' ')}');
    _appendLog('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

    if (!await backendDir.exists()) {
      final err = 'Backend directory not found: ${backendDir.path}';
      _appendLog('❌ ERROR: $err');
      _status = BackendStatus.failed;
      _lastError = err;
      notifyListeners();
      return false;
    }

    final scriptFile = File('${backendDir.path}${Platform.pathSeparator}${_mode.script}');
    if (!await scriptFile.exists()) {
      final err = 'Script not found: ${scriptFile.path}';
      _appendLog('❌ ERROR: $err');
      _status = BackendStatus.failed;
      _lastError = err;
      notifyListeners();
      return false;
    }

    try {
      // Prepend the bundled interpreter's own folders to PATH so that if
      // anything in the process tree (a library, a future code path) shells
      // out to a bare "python" rather than sys.executable, it resolves to
      // THIS interpreter — not whatever "python" happens to be first on the
      // machine's own PATH (e.g. a system install with different package
      // versions). Keeps the standalone build self-contained end to end.
      final pyDir = File(pythonExe).parent.path;      // .../runtime/Scripts
      final pyRoot = Directory(pyDir).parent.path;     // .../runtime
      final path = Platform.environment['Path'] ?? Platform.environment['PATH'] ?? '';
      final env = Map<String, String>.from(Platform.environment)
        ..['PYTHONUNBUFFERED'] = '1'
        ..['PYTHONIOENCODING'] = 'utf-8'
        ..['PYTHONUTF8'] = '1'
        ..['PATH'] = '$pyDir;$pyRoot;$path'
        ..['Path'] = '$pyDir;$pyRoot;$path';

      // The bundled runtime carries its own copy of the standard library +
      // runtime DLLs (backend/runtime/Lib, /DLLs — see backend/README.md) so
      // it needs nothing from any system-wide Python install. PYTHONHOME
      // tells the interpreter to resolve the stdlib from there instead of
      // wherever pyvenv.cfg's `home` points. Only set this for that bundled
      // layout — forcing it onto an arbitrary system `python` (the dev
      // fallback, pythonPath: "python") would break it outright.
      if (Directory('$pyRoot${Platform.pathSeparator}Lib').existsSync() &&
          Directory('$pyRoot${Platform.pathSeparator}DLLs').existsSync()) {
        env['PYTHONHOME'] = pyRoot;
      }

      // No shell wrapper: `pythonExe` is an absolute path that (for the
      // bundled build) contains a space — `cmd.exe /c "<quoted path>" args`
      // is a well-known Windows quoting minefield and was observed to
      // mis-launch a *second*, wrong (system) interpreter as a child.
      // Process.start resolves a bare "python" via PATH on Windows even
      // without runInShell, so the dev fallback (pythonPath: "python") still
      // works with this off.
      final proc = await Process.start(
        pythonExe,
        [_mode.script, ..._mode.args],
        workingDirectory: backendDir.path,
        environment: env,
      );

      _process = proc;
      _pid = proc.pid;
      _startTime = DateTime.now();
      _appendLog('✔ Process spawned with PID: $_pid');

      // Stream stdout & stderr
      proc.stdout
          .transform(utf8.decoder)
          .listen((data) => _appendLog(data), onError: (e) => _appendLog('stdout error: $e'));

      proc.stderr
          .transform(utf8.decoder)
          .listen((data) => _appendLog(data), onError: (e) => _appendLog('stderr error: $e'));

      proc.exitCode.then((code) {
        _appendLog('⏹ Process exited with code $code');
        if (_status != BackendStatus.stopping) {
          _status = code == 0 ? BackendStatus.stopped : BackendStatus.failed;
          if (code != 0) {
            _lastError = 'Backend process exited with code $code';
          }
        } else {
          _status = BackendStatus.stopped;
        }
        _process = null;
        _pid = null;
        _startTime = null;
        notifyListeners();
      });

      // Poll server readiness for up to 30 seconds
      final ready = await _waitForServerReady(const Duration(seconds: 30));
      if (ready) {
        _status = BackendStatus.running;
        _lastError = null;
        _appendLog('✔ IBVAP Server is ONLINE and accepting requests!');
      } else if (_status == BackendStatus.starting) {
        _status = BackendStatus.failed;
        _lastError = 'Server did not respond within timeout';
        _appendLog('⚠️ Server start timed out — check console output above.');
      }
      notifyListeners();
      return _status == BackendStatus.running;
    } catch (e) {
      _status = BackendStatus.failed;
      _lastError = e.toString();
      _appendLog('❌ Failed to start process: $e');
      _process = null;
      _pid = null;
      _startTime = null;
      notifyListeners();
      return false;
    }
  }

  /// Stop the backend — the child process this app launched, or (when the
  /// console attached to one already running) the external server via its
  /// shutdown API. Returns true only once the server has stopped answering.
  Future<bool> stop() async {
    final ownProcess = _process != null;
    if (!ownProcess && !await _serverAnswers()) {
      _status = BackendStatus.stopped;
      notifyListeners();
      return true;
    }

    _status = BackendStatus.stopping;
    notifyListeners();
    _appendLog(ownProcess
        ? '⏹ Stopping IBVAP Backend (PID: $_pid)…'
        : '⏹ Requesting shutdown of external IBVAP Backend at ${AppConfig.instance.baseUrl}…');

    // 1. Graceful HTTP shutdown first (flushes the event store / hash chain).
    try {
      final client = http.Client();
      final headers = <String, String>{};
      final tok = AppConfig.instance.effectiveApiToken;
      if (tok != null && tok.isNotEmpty) headers['X-IBVAP-Token'] = tok;
      final res = await client
          .post(AppConfig.instance.endpoint('/api/shutdown'), headers: headers)
          .timeout(const Duration(milliseconds: 1500));
      client.close();
      if (res.statusCode == 401 || res.statusCode == 403) {
        _appendLog('⚠️ Shutdown refused by server: API token required.');
      }
    } catch (_) {}

    // 2. Hard-kill our own process tree if it hasn't exited.
    if (ownProcess) {
      await Future.delayed(const Duration(milliseconds: 600));
      if (_pid != null) {
        if (Platform.isWindows) {
          try {
            await Process.run('taskkill', ['/F', '/T', '/PID', '$_pid']);
          } catch (_) {}
        } else {
          _process?.kill(ProcessSignal.sigterm);
        }
      }
      _process?.kill(ProcessSignal.sigkill);
    }

    // 3. Verify. Reporting "stopped" while cameras are still being processed
    //    (or while they're not, but we claim they are) misleads the operator.
    final down = await _waitForServerDown(const Duration(seconds: 5));
    _process = null;
    _pid = null;
    _startTime = null;
    if (down) {
      _status = BackendStatus.stopped;
      _lastError = null;
      _appendLog('✔ IBVAP Backend stopped.');
    } else {
      _status = BackendStatus.failed;
      _lastError = 'Backend still answering after shutdown request';
      _appendLog('⚠️ Server is still responding — it may be owned by another '
          'process or refused the shutdown. Stop it manually.');
    }
    notifyListeners();
    return down;
  }

  Future<bool> _serverAnswers() async {
    final client = http.Client();
    try {
      final res = await client
          .get(AppConfig.instance.endpoint('/status'))
          .timeout(const Duration(milliseconds: 700));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    } finally {
      client.close();
    }
  }

  Future<bool> _waitForServerDown(Duration timeout) async {
    final deadline = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(deadline)) {
      if (!await _serverAnswers()) return true;
      await Future.delayed(const Duration(milliseconds: 400));
    }
    return !await _serverAnswers();
  }

  /// Restart the backend process.
  Future<bool> restart({BackendMode? mode}) async {
    await stop();
    await Future.delayed(const Duration(milliseconds: 800));
    return start(mode: mode);
  }

  /// Probes `/status` until 200 OK or timeout.
  Future<bool> _waitForServerReady(Duration timeout) async {
    final deadline = DateTime.now().add(timeout);
    final url = AppConfig.instance.endpoint('/status');
    final client = http.Client();

    while (DateTime.now().isBefore(deadline)) {
      if (_process == null || _status == BackendStatus.failed) {
        client.close();
        return false;
      }
      try {
        final res = await client.get(url).timeout(const Duration(milliseconds: 600));
        if (res.statusCode == 200) {
          client.close();
          return true;
        }
      } catch (_) {}
      await Future.delayed(const Duration(milliseconds: 500));
    }
    client.close();
    return false;
  }

  bool _disposed = false;

  @override
  void notifyListeners() {
    if (!_disposed) super.notifyListeners();
  }

  /// Tears down only a backend this console launched. An external backend is
  /// left running: closing a console must not take surveillance offline.
  @override
  void dispose() {
    _disposed = true;
    final pid = _pid;
    if (pid != null) {
      if (Platform.isWindows) {
        Process.run('taskkill', ['/F', '/T', '/PID', '$pid']).ignore();
      } else {
        _process?.kill(ProcessSignal.sigterm);
      }
    }
    logRevision.dispose();
    super.dispose();
  }
}
