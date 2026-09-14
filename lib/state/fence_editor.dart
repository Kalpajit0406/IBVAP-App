import 'package:flutter/foundation.dart';

import '../api/ibvap_client.dart';
import '../models/fence.dart';
import '../util/describe_error.dart';

/// Holds the virtual-fence list and the in-progress draw / edit draft. Shared
/// by the Monitor mosaic (where points are placed) and the Fences sidebar
/// (the form + list), so both stay in sync.
class FenceEditor extends ChangeNotifier {
  FenceEditor(this._client);

  final IbvapClient _client;

  List<Fence> _fences = [];
  List<Fence> get fences => List.unmodifiable(_fences);

  bool _loading = false;
  bool get loading => _loading;
  String? error;
  bool _busy = false;
  bool get busy => _busy;

  // ── draft ─────────────────────────────────────────────────────────────
  bool armed = false;
  String? editingId;
  int? draftCam;
  String kind = 'polygon'; // 'polygon' | 'line'
  String targets = 'any'; // 'any' | 'person' | 'vehicle'
  String direction = 'both'; // 'both' | 'a2b' | 'b2a'
  String label = '';
  final List<(double, double)> draft = [];

  int get _need => kind == 'line' ? 2 : 3;
  bool get canSave => armed && draft.length >= _need;

  int fenceCountForCam(int cam) => _fences.where((f) => f.camId == cam).length;

  // ── loads ─────────────────────────────────────────────────────────────
  Future<void> load() async {
    _loading = true;
    error = null;
    notifyListeners();
    try {
      final rows = await _client.fences();
      _fences = rows
          .whereType<Map>()
          .map((e) => Fence.fromJson(e.cast<String, dynamic>()))
          .toList();
    } catch (e) {
      error = describeError(e);
    }
    _loading = false;
    notifyListeners();
  }

  // ── draw / edit lifecycle ────────────────────────────────────────────
  void startDraw(int cam, String kind) {
    armed = true;
    editingId = null;
    draftCam = cam;
    this.kind = kind;
    draft.clear();
    notifyListeners();
  }

  void startEdit(String id) {
    final f = _fences.where((x) => x.id == id).firstOrNull;
    if (f == null) return;
    armed = true;
    editingId = id;
    draftCam = f.camId;
    kind = f.kind;
    targets = f.targets.isNotEmpty ? f.targets.first : 'any';
    direction = f.direction;
    label = f.label;
    draft
      ..clear()
      ..addAll(f.points);
    notifyListeners();
  }

  void addPoint(int cam, double u, double v) {
    if (!armed || cam != draftCam) return;
    draft.add((u.clamp(0, 1), v.clamp(0, 1)));
    notifyListeners();
  }

  void undoPoint() {
    if (draft.isNotEmpty) {
      draft.removeLast();
      notifyListeners();
    }
  }

  void cancel() {
    armed = false;
    editingId = null;
    draftCam = null;
    draft.clear();
    label = '';
    notifyListeners();
  }

  void setKind(String k) {
    if (armed) return;
    kind = k;
    notifyListeners();
  }

  void setTargets(String t) {
    targets = t;
    notifyListeners();
  }

  void setDirection(String d) {
    direction = d;
    notifyListeners();
  }

  void setLabel(String l) {
    label = l;
  }

  // ── writes ────────────────────────────────────────────────────────────
  Future<String?> save() async {
    if (!canSave || draftCam == null) return 'need $_need+ points';
    final draftFence = Fence(
      id: editingId,
      camId: draftCam!,
      kind: kind,
      points: List.of(draft),
      direction: kind == 'line' ? direction : 'both',
      targets: [targets],
      label: label.trim(),
      enabled: true,
    );
    final all = <Fence>[
      for (final f in _fences)
        if (f.id != editingId) f,
      draftFence,
    ];
    final err = await _postAll(all);
    if (err == null) {
      cancel();
      await load();
    }
    return err;
  }

  Future<String?> setEnabled(String id, bool enabled) async {
    final all = _fences
        .map((f) => f.id == id ? f.copyWith(enabled: enabled) : f)
        .toList();
    final err = await _postAll(all);
    if (err == null) await load();
    return err;
  }

  Future<String?> delete(String id) async {
    _busy = true;
    notifyListeners();
    String? err;
    try {
      await _client.deleteFence(id);
    } on IbvapApiException catch (e) {
      err = e.message;
    }
    _busy = false;
    if (err == null) {
      await load();
    } else {
      notifyListeners();
    }
    return err;
  }

  Future<String?> clearCam(int cam) async {
    final all = _fences.where((f) => f.camId != cam).toList();
    final err = await _postAll(all);
    if (err == null) await load();
    return err;
  }

  Future<String?> _postAll(List<Fence> all) async {
    _busy = true;
    notifyListeners();
    String? err;
    try {
      await _client.saveFences(all.map((f) => f.toJson()).toList());
    } catch (e) {
      err = describeError(e);
    }
    _busy = false;
    notifyListeners();
    return err;
  }
}
