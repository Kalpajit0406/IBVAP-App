import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../api/ibvap_client.dart';
import '../config/app_config.dart';
import '../models/server_status.dart';
import '../models/threat.dart';
import '../services/backend_manager.dart';
import '../services/tunnel_manager.dart';
import '../util/describe_error.dart';
import 'alert_log.dart';
import 'fence_editor.dart';

/// * [connecting] — no successful poll yet since start / server change.
/// * [online] — the last poll succeeded.
/// * [degraded] — recent polls failed but not long enough to call the link
///   down; the last good picture is still shown, flagged as stale.
/// * [offline] — [AppState.offlineAfterFailures] consecutive failures *and*
///   no good poll for [AppState.offlineAfter].
enum LinkState { connecting, online, degraded, offline }

/// Owns the connection to the IBVAP server: polls `/status` (and, more slowly,
/// `/api/models`, `/api/snapshots`, `/api/anpr/results`) and broadcasts the
/// latest [ServerStatus], a [LinkState] and the derived [ThreatPicture].
class AppState extends ChangeNotifier {
  AppState({IbvapClient? client, BackendManager? backend})
      : _client = client ?? IbvapClient(),
        alerts = AlertLog(),
        tunnel = TunnelManager(),
        backend = backend ?? BackendManager() {
    fences = FenceEditor(_client);
    this.backend.addListener(_onBackendChanged);
    alerts.addListener(_onAlertsChanged);
  }

  static const offlineAfterFailures = 3;
  static const offlineAfter = Duration(seconds: 4);
  static const staleAfter = Duration(seconds: 3);

  final IbvapClient _client;
  IbvapClient get client => _client;

  final AlertLog alerts;
  final BackendManager backend;
  final TunnelManager tunnel;
  late final FenceEditor fences;

  Timer? _statusTimer;
  Timer? _slowTimer;
  Timer? _alarmTimer;
  int _slowTick = 0;
  bool _tickBusy = false;
  bool _slowBusy = false;
  int _generation = 0; // bumped on server change; late replies are discarded

  LinkState _link = LinkState.connecting;
  LinkState get link => _link;
  bool get linkUp => _link == LinkState.online;

  ServerStatus? _status;
  ServerStatus? get status => _status;

  ThreatPicture _threat = ThreatPicture.unknown;
  ThreatPicture get threat => _threat;

  Map<String, dynamic> _models = const {};
  Map<String, dynamic> get models => _models;

  String? _lastError;
  String? get lastError => _lastError;

  DateTime? _lastOk;
  DateTime? get lastOk => _lastOk;

  int _consecutiveFailures = 0;
  int get consecutiveFailures => _consecutiveFailures;

  /// Round-trip time of the last successful `/status` poll.
  Duration? _latency;
  Duration? get latency => _latency;

  /// How old the displayed picture is; null before the first good poll.
  Duration? get dataAge =>
      _lastOk == null ? null : DateTime.now().difference(_lastOk!);

  bool get isStale {
    final a = dataAge;
    return a == null || a > staleAfter;
  }

  String get baseUrl => AppConfig.instance.baseUrl;
  int get pollIntervalMs => AppConfig.instance.pollIntervalMs;

  bool get isRetraining =>
      (_status?.learning['last_train'] as Map?)?['state'] == 'running';

  bool _started = false;
  bool _disposed = false;

  @override
  void notifyListeners() {
    if (!_disposed) super.notifyListeners();
  }

  void start() {
    _started = true;
    _restartTimers();
    _tick();
    _slowRefresh();
    fences.load();
  }

  void stop() {
    _started = false;
    _statusTimer?.cancel();
    _slowTimer?.cancel();
    _alarmTimer?.cancel();
    _statusTimer = null;
    _slowTimer = null;
    _alarmTimer = null;
  }

  void _restartTimers() {
    _statusTimer?.cancel();
    _slowTimer?.cancel();
    _statusTimer = Timer.periodic(
        Duration(milliseconds: pollIntervalMs), (_) => _tick());
    _slowTimer =
        Timer.periodic(const Duration(seconds: 2), (_) => _slowRefresh());
  }

  BackendStatus? _seenBackendStatus;
  int? _seenBackendPid;

  // BackendManager notifies on every stdout line. Only lifecycle changes
  // matter to the console; forwarding log chatter rebuilt the whole shell —
  // video stage included — at the backend's logging rate.
  void _onBackendChanged() {
    if (backend.status == _seenBackendStatus && backend.pid == _seenBackendPid) {
      return;
    }
    _seenBackendStatus = backend.status;
    _seenBackendPid = backend.pid;
    if (backend.isRunning && _link != LinkState.online) refreshNow();
    notifyListeners();
  }

  // ── alarm cadence ──────────────────────────────────────────────────────
  // Re-sound every few seconds while anything critical is unacknowledged, so
  // an operator who looked away is still pulled back to the console.
  void _onAlertsChanged() {
    final pending = alerts.unackedCount > 0;
    if (pending && _alarmTimer == null) {
      _soundAlarm();
      _alarmTimer = Timer.periodic(const Duration(seconds: 6), (_) {
        if (alerts.unackedCount == 0) {
          _alarmTimer?.cancel();
          _alarmTimer = null;
        } else {
          _soundAlarm();
        }
      });
    } else if (!pending) {
      _alarmTimer?.cancel();
      _alarmTimer = null;
    }
    notifyListeners();
  }

  void _soundAlarm() {
    if (!_started || !AppConfig.instance.alarmSound) return;
    SystemSound.play(SystemSoundType.alert);
  }

  /// Point the client at a new server and re-poll immediately.
  Future<void> setBaseUrl(String value) async {
    await AppConfig.instance.setBaseUrl(value);
    _generation++;
    _link = LinkState.connecting;
    _status = null;
    _threat = ThreatPicture.unknown;
    _models = const {};
    _lastError = null;
    _lastOk = null;
    _latency = null;
    _consecutiveFailures = 0;
    alerts.clear();
    notifyListeners();
    await _tick(force: true);
    await _slowRefresh(force: true);
    await fences.load();
  }

  Future<void> setPollInterval(int ms) async {
    await AppConfig.instance.setPollIntervalMs(ms);
    if (_started) _restartTimers();
    notifyListeners();
  }

  Future<void> refreshNow() async {
    await _tick(force: true);
    await _slowRefresh(force: true);
  }

  // ── polls ──────────────────────────────────────────────────────────────
  // Timer.periodic doesn't wait for an async callback, so without the busy
  // flags a slow server (5 s timeout vs a 1 s poll) stacks up to five
  // concurrent /status requests and replies can land out of order.
  Future<void> _tick({bool force = false}) async {
    if (_tickBusy && !force) return;
    _tickBusy = true;
    final gen = _generation;
    final t0 = DateTime.now();
    try {
      final raw = await _client.status();
      if (gen != _generation) return;
      final s = ServerStatus(raw);
      _status = s;
      _latency = DateTime.now().difference(t0);
      _link = LinkState.online;
      _lastError = null;
      _lastOk = DateTime.now();
      _consecutiveFailures = 0;
      _threat = ThreatPicture.from(s, linkUp: true);
      alerts.ingestStatus(s);
    } catch (e) {
      if (gen != _generation) return;
      _consecutiveFailures++;
      _lastError = describeError(e);
      final age = dataAge;
      final down = _consecutiveFailures >= offlineAfterFailures &&
          (age == null || age >= offlineAfter);
      _link = _lastOk == null
          ? (_consecutiveFailures >= offlineAfterFailures
              ? LinkState.offline
              : LinkState.connecting)
          : (down ? LinkState.offline : LinkState.degraded);
      _threat = ThreatPicture.unknown;
    } finally {
      if (gen == _generation) _tickBusy = false;
      if (gen == _generation) notifyListeners();
    }
  }

  Future<void> _slowRefresh({bool force = false}) async {
    if (_link == LinkState.offline || _link == LinkState.connecting) {
      if (!force) return;
    }
    if (_slowBusy && !force) return;
    _slowBusy = true;
    final gen = _generation;
    _slowTick++;
    try {
      if (_slowTick % 5 == 1 || _models.isEmpty) {
        final m = await _client.models();
        if (gen != _generation) return;
        _models = m;
      }
      final snaps = await _client.snapshots(limit: 20);
      if (gen != _generation) return;
      alerts.ingestSnapshots(snaps);
      final anprResults = await _client.anprResults(limit: 20);
      if (gen != _generation) return;
      alerts.ingestAnprResults(anprResults);
      final faceResults = await _client.faceResults(limit: 20);
      if (gen != _generation) return;
      alerts.ingestFaceResults(faceResults);
      notifyListeners();
    } catch (_) {
      // non-fatal — the /status poll owns the link state
    } finally {
      if (gen == _generation) _slowBusy = false;
    }
  }

  @override
  void dispose() {
    _disposed = true;
    _generation++;
    stop();
    backend.removeListener(_onBackendChanged);
    alerts.removeListener(_onAlertsChanged);
    backend.dispose();
    tunnel.dispose();
    alerts.dispose();
    fences.dispose();
    _client.close();
    super.dispose();
  }
}
