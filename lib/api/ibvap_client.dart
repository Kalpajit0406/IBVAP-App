import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/app_config.dart';

class IbvapApiException implements Exception {
  IbvapApiException(this.message, {this.statusCode, this.cause});
  final String message;
  final int? statusCode;

  /// The underlying transport error, when the request never got a response.
  final Object? cause;

  bool get isAuth => statusCode == 401 || statusCode == 403;

  @override
  String toString() => 'IbvapApiException($statusCode): $message';
}

/// Thin wrapper over the IBVAP server HTTP API.
///
/// Every method reads the base URL from [AppConfig] at call time, so changing
/// the server address takes effect on the next request with no re-wiring.
class IbvapClient {
  IbvapClient({http.Client? client, this.timeout = const Duration(seconds: 5)})
      : _client = client ?? http.Client();

  final http.Client _client;
  final Duration timeout;

  AppConfig get _cfg => AppConfig.instance;

  // ── reads ────────────────────────────────────────────────────────────────
  Future<Map<String, dynamic>> status() => _getJson('/status');

  Future<Map<String, dynamic>> devices() => _getJson('/devices');

  Future<Map<String, dynamic>> models() => _getJson('/api/models');

  Future<Map<String, dynamic>> meta(int camId) => _getJson('/meta/$camId');

  Future<List<dynamic>> fences({int? camId}) async {
    final path = camId == null ? '/api/fences' : '/api/fences?cam_id=$camId';
    final j = await _getJson(path);
    return (j['fences'] as List<dynamic>?) ?? const [];
  }

  Future<List<dynamic>> snapshots({int? camId, int limit = 30}) async {
    final q = StringBuffer('/api/snapshots?limit=$limit');
    if (camId != null) q.write('&cam_id=$camId');
    final j = await _getJson(q.toString());
    return (j['snapshots'] as List<dynamic>?) ?? const [];
  }

  /// Event-triggered ANPR results: one picture + plate + vehicle type per
  /// vehicle arrival, from data/anpr_results/ (separate from /api/snapshots).
  Future<List<dynamic>> anprResults({int? camId, int limit = 30}) async {
    final q = StringBuffer('/api/anpr/results?limit=$limit');
    if (camId != null) q.write('&cam_id=$camId');
    final j = await _getJson(q.toString());
    return (j['results'] as List<dynamic>?) ?? const [];
  }

  Future<Map<String, dynamic>> learnPool({String? status}) {
    final p =
        status == null ? '/api/learn/pool' : '/api/learn/pool?status=$status';
    return _getJson(p);
  }

  Future<Map<String, dynamic>> streams() => _getJson('/api/streams');

  Future<Map<String, dynamic>> mobileInfo() => _getJson('/api/mobile-info');

  Future<Map<String, dynamic>> upsertStream(Map<String, dynamic> body) =>
      _postJson('/api/streams', body);

  Future<Map<String, dynamic>> deleteStream(int camId) =>
      _sendJson('DELETE', '/api/streams/$camId');

  Future<Map<String, dynamic>> probeStream(String url, {String transport = 'tcp'}) =>
      _postJson('/api/streams/probe', {'url': url, 'transport': transport});

  Future<Map<String, dynamic>> discoverLanCameras() =>
      _postJson('/api/streams/discover', const {});

  // ── writes ──────────────────────────────────────────────────────────────
  Future<Map<String, dynamic>> setLayout({
    required String mode, // 'grid' | 'focus'
    List<int> mains = const [],
  }) =>
      _postJson('/api/layout', {'mode': mode, 'mains': mains});

  Future<Map<String, dynamic>> saveFences(List<dynamic> fences) =>
      _postJson('/api/fences', {'fences': fences});

  Future<Map<String, dynamic>> deleteFence(String id) =>
      _sendJson('DELETE', '/api/fences/${Uri.encodeComponent(id)}');

  Future<Map<String, dynamic>> switchModel(Map<String, dynamic> body) =>
      _postJson('/api/switch-model', body);

  Future<Map<String, dynamic>> reconnectCamera(int camId) =>
      _postJson('/api/reconnect/$camId', const {});

  Future<Map<String, dynamic>> learnReview({
    required String id,
    required String verdict, // keep | drop | background
    List<dynamic>? boxes,
  }) {
    final body = <String, dynamic>{'id': id, 'verdict': verdict};
    if (boxes != null) body['boxes'] = boxes;
    return _postJson('/api/learn/review', body);
  }

  Future<Map<String, dynamic>> learnCalibration(Map<String, dynamic> body) =>
      _postJson('/api/learn/calibration', body);

  Future<Map<String, dynamic>> learnRetrain({
    String mode = 'detector', // detector | posture | anpr | thermal | weapon
    bool dryRun = false,
    int? epochs,
    int? batch,
    int? freeze,
  }) {
    final body = <String, dynamic>{'mode': mode, 'dry_run': dryRun};
    if (epochs != null) body['epochs'] = epochs;
    if (batch != null) body['batch'] = batch;
    if (freeze != null) body['freeze'] = freeze;
    return _postJson('/api/learn/retrain', body);
  }

  Future<Map<String, dynamic>> shutdown() =>
      _postJson('/api/shutdown', const {});

  // ── evidence / incident history ─────────────────────────────────────────
  /// Walks the whole SHA-256 chain server-side; can take a moment on a long log.
  Future<Map<String, dynamic>> verifyEvidence() =>
      _getJson('/api/evidence/verify', timeout: const Duration(seconds: 30));

  Future<List<dynamic>> evidenceRecent({int limit = 50}) async {
    final j = await _getJson('/api/evidence/recent?limit=$limit');
    return (j['records'] as List<dynamic>?) ?? const [];
  }

  Future<Map<String, dynamic>> events({
    int limit = 200,
    int? camId,
    String? level,
    double? since,
  }) {
    final q = StringBuffer('/api/events?limit=$limit');
    if (camId != null) q.write('&cam_id=$camId');
    if (level != null) q.write('&level=${Uri.encodeQueryComponent(level)}');
    if (since != null) q.write('&since=$since');
    return _getJson(q.toString());
  }

  // ── URL helpers (for viewers / Image.network) ───────────────────────────
  /// Composited MJPEG mosaic of every connected camera.
  Uri get streamUrl => _cfg.endpoint('/stream');

  /// Single-camera MJPEG (debug endpoint on the server).
  Uri cameraStreamUrl(int camId) => _cfg.endpoint('/stream/$camId');

  /// A saved snapshot JPEG. [file] is the `file` / `raw_file` value from
  /// `/api/snapshots` (already `"<cam_id>/<name>.jpg"`).
  Uri snapshotUrl(String file) => _cfg.endpoint('/snap/$file');

  /// A continuous-learning candidate thumbnail.
  Uri learnThumbUrl(String itemId) => _cfg.endpoint('/learn/thumb/$itemId.jpg');

  /// An ANPR-result JPEG. [file] is the `file` value from
  /// `/api/anpr/results` (already `"<cam_id>/<name>.jpg"`).
  Uri anprSnapUrl(String file) => _cfg.endpoint('/anpr_snap/$file');

  // ── plumbing ────────────────────────────────────────────────────────────
  Map<String, String> _headers({bool json = false}) {
    final h = <String, String>{'Accept': 'application/json'};
    if (json) h['Content-Type'] = 'application/json';
    final tok = _cfg.effectiveApiToken;
    if (tok != null && tok.isNotEmpty) h['X-IBVAP-Token'] = tok;
    return h;
  }

  Future<Map<String, dynamic>> _getJson(String path, {Duration? timeout}) async {
    final res = await _guard(() => _client
        .get(_cfg.endpoint(path), headers: _headers())
        .timeout(timeout ?? this.timeout));
    return _decodeObject(res);
  }

  Future<Map<String, dynamic>> _postJson(
          String path, Map<String, dynamic> body) =>
      _sendJson('POST', path, body);

  Future<Map<String, dynamic>> _sendJson(
    String method,
    String path, [
    Map<String, dynamic>? body,
  ]) async {
    final req = http.Request(method, _cfg.endpoint(path))
      ..headers.addAll(_headers(json: true));
    if (body != null) req.body = jsonEncode(body);
    // The timeout must cover the body too — send() resolves on headers alone,
    // so a server that stalls mid-body would otherwise hang the call forever.
    final res = await _guard(() async =>
        http.Response.fromStream(await _client.send(req).timeout(timeout))
            .timeout(timeout));
    return _decodeObject(res);
  }

  Future<http.Response> _guard(Future<http.Response> Function() run) async {
    try {
      return await run();
    } on IbvapApiException {
      rethrow;
    } catch (e) {
      throw IbvapApiException('Cannot reach ${_cfg.baseUrl}', cause: e);
    }
  }

  Map<String, dynamic> _decodeObject(http.Response res) {
    if (res.statusCode >= 400) {
      throw IbvapApiException(_errText(res), statusCode: res.statusCode);
    }
    if (res.body.isEmpty) return const {};
    final Object? decoded;
    try {
      decoded = jsonDecode(res.body);
    } on FormatException catch (e) {
      throw IbvapApiException('Malformed response from server',
          statusCode: res.statusCode, cause: e);
    }
    if (decoded is Map<String, dynamic>) return decoded;
    return {'data': decoded};
  }

  String _errText(http.Response res) {
    try {
      final j = jsonDecode(res.body);
      if (j is Map && j['error'] != null) return j['error'].toString();
      if (j is Map && j['errors'] != null) {
        return (j['errors'] as List).join('; ');
      }
    } catch (_) {}
    return 'HTTP ${res.statusCode}';
  }

  void close() => _client.close();
}
