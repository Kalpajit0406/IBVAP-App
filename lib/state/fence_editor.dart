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
  String severity = 'Critical'; // Fence.severities
  // The three free-text settings are held as typed and only interpreted at
  // save time, so half-typed input ("2", "22:") is never mis-parsed mid-edit.
  String loiterText = ''; // seconds; blank / 0 = off
  String armedFrom = ''; // "HH:MM" local time; blank = always armed
  String armedTo = '';
  String inbound = ''; // '' | 'a2b' | 'b2a'  (lines only)
  String followText = ''; // seconds; blank / 0 = off  (lines only)
  final List<(double, double)> draft = [];

  /// Bumped whenever a draft is loaded, started or cancelled. The form's text
  /// fields watch it to know when to overwrite what they show; comparing the
  /// text itself instead would fight the user while they type.
  int revision = 0;

  static final _hhmm = RegExp(r'^([01]\d|2[0-3]):[0-5]\d$');

  int get _need => kind == 'line' ? 2 : 3;

  /// Why the behaviour settings cannot be saved, or null when they are fine.
  /// Mirrors the server's validation so the operator finds out here rather
  /// than from a rejected request.
  String? get settingsError {
    final t = loiterText.trim();
    if (t.isNotEmpty) {
      final v = double.tryParse(t);
      if (v == null || v < 0 || v > 86400) {
        return 'Loiter time must be 0–86400 seconds';
      }
    }
    final f = followText.trim();
    if (f.isNotEmpty) {
      final v = double.tryParse(f);
      if (v == null || v < 0 || v > 600) {
        return 'Follow window must be 0–600 seconds';
      }
    }
    final from = armedFrom.trim(), to = armedTo.trim();
    if (from.isEmpty != to.isEmpty) return 'Set both arming times, or neither';
    for (final v in [from, to]) {
      if (v.isNotEmpty && !_hhmm.hasMatch(v)) {
        return 'Arming times must be 24-hour HH:MM';
      }
    }
    return null;
  }

  bool get canSave => armed && draft.length >= _need && settingsError == null;

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
  void _resetSettings() {
    label = '';
    severity = 'Critical';
    loiterText = '';
    armedFrom = '';
    armedTo = '';
    inbound = '';
    followText = '';
  }

  /// Seconds as the operator would type them: 30.0 -> "30", 0 -> blank.
  static String _secondsText(double v) => v <= 0
      ? ''
      : (v == v.roundToDouble() ? v.round().toString() : v.toString());

  void startDraw(int cam, String kind) {
    armed = true;
    editingId = null;
    draftCam = cam;
    this.kind = kind;
    draft.clear();
    // Deliberately NOT resetting the settings: the form is filled in first and
    // Draw pressed second, so what the operator already chose must survive it.
    // Only cancel() and a completed save() clear it.
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
    severity = Fence.severities.contains(f.severity) ? f.severity : 'Critical';
    // 30.0 shows as "30"; 0 shows as blank rather than a stray "0".
    loiterText = _secondsText(f.loiterAfterS);
    followText = _secondsText(f.followWindowS);
    armedFrom = f.armedFrom;
    armedTo = f.armedTo;
    inbound = f.inbound;
    draft
      ..clear()
      ..addAll(f.points);
    revision++;
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
    _resetSettings();
    revision++;
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

  void setSeverity(String s) {
    severity = s;
    notifyListeners();
  }

  void setInbound(String i) {
    inbound = i;
    notifyListeners();
  }

  // Text settings notify so the inline validation message and the Save button
  // track each keystroke; the form's controllers are not reset by this, only by
  // a change of [revision].
  void setLoiterText(String t) {
    loiterText = t;
    notifyListeners();
  }

  void setFollowText(String t) {
    followText = t;
    notifyListeners();
  }

  void setArmedFrom(String t) {
    armedFrom = t;
    notifyListeners();
  }

  void setArmedTo(String t) {
    armedTo = t;
    notifyListeners();
  }

  // ── writes ────────────────────────────────────────────────────────────
  Future<String?> save() async {
    if (!canSave || draftCam == null) {
      return settingsError ?? 'need $_need+ points';
    }
    // Start from the fence being edited, not a blank one: copyWith carries its
    // `extra` (server-owned created_at, and any field this client does not know
    // yet) through the save. Building a fresh Fence here is what used to erase
    // everything not on the form.
    final existing = editingId == null
        ? null
        : _fences.where((x) => x.id == editingId).firstOrNull;
    final draftFence = (existing ??
            Fence(id: editingId, camId: draftCam!, kind: kind, points: const []))
        .copyWith(
      camId: draftCam!,
      kind: kind,
      points: List.of(draft),
      direction: kind == 'line' ? direction : 'both',
      targets: [targets],
      label: label.trim(),
      enabled: true,
      severity: severity,
      loiterAfterS: double.tryParse(loiterText.trim()) ?? 0,
      followWindowS: double.tryParse(followText.trim()) ?? 0,
      armedFrom: armedFrom.trim(),
      armedTo: armedTo.trim(),
      inbound: inbound,
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
