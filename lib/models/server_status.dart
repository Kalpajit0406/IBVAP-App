import 'fence.dart';

/// A light view over the IBVAP `/status` payload. Only the fields the client
/// renders are lifted out; [raw] keeps the whole map for anything else.
class ServerStatus {
  ServerStatus(this.raw);

  final Map<String, dynamic> raw;

  String get device => (raw['device'] ?? 'unknown').toString();
  bool get onGpu => device.isNotEmpty && device != 'cpu' && device != 'unknown';
  String get activeModel => (raw['active_model'] ?? '—').toString();
  bool get switching => raw['switching'] == true;
  String get inputMode => (raw['input_mode'] ?? '—').toString();

  double get fps => _d(raw['fps']);
  double get inferenceMs => _d(raw['inference_ms']);
  int get framesProcessed => _i(raw['frames_processed']);

  Map<String, dynamic> get pipeline => _m(raw['pipeline']);
  double get gpuSavingPct => _d(pipeline['gpu_saving_pct']);
  double get meanBatch => _d(pipeline['mean_batch']);
  double get muxerTps => _d(pipeline['muxer_tps']);
  double get workerBps => _d(pipeline['worker_bps']);
  int get snapshotsDropped => _i(pipeline['snapshots_dropped']);

  List<Map<String, dynamic>> get devices => ((raw['devices'] as List?) ?? const [])
      .whereType<Map>()
      .map((e) => e.cast<String, dynamic>())
      .toList();

  int get cameraCount => devices.length;
  int get camerasLive => devices.where((d) => (d['state'] ?? '') == 'live').length;

  Map<String, dynamic> get meta => _m(raw['meta']);
  Map<String, dynamic> metaFor(int camId) => _m(meta['$camId']);

  Map<String, dynamic> get geofence => _m(raw['geofence']);
  Map<String, dynamic> get snapshots => _m(raw['snapshots']);
  Map<String, dynamic> get anpr => _m(raw['anpr']);
  Map<String, dynamic> get weapon => _m(raw['weapon']);
  Map<String, dynamic> get learning => _m(raw['learning']);
  Map<String, dynamic> get mosaic => _m(raw['mosaic']);

  MosaicLayout get layout => MosaicLayout.fromStatus(mosaic);

  /// cam ids currently producing frames, from `/status.mosaic.ids`.
  List<int> get cameraIds => ((mosaic['ids'] as List?) ?? const [])
      .map((e) => (e as num).toInt())
      .toList();

  /// {cam_id: {fence_id: bool}} — which fences are breaching right now.
  Map<String, dynamic> get geofenceActive => _m(geofence['active']);

  bool camBreaching(int camId) {
    final z = _m(geofenceActive['$camId']);
    return z.values.any((v) => v == true);
  }

  static Map<String, dynamic> _m(Object? v) =>
      v is Map ? v.cast<String, dynamic>() : const {};
  static double _d(Object? v) => v is num ? v.toDouble() : 0.0;
  static int _i(Object? v) => v is num ? v.toInt() : 0;
}
