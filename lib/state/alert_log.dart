import 'package:flutter/foundation.dart';

import '../models/alert.dart';
import '../models/server_status.dart';

/// Client-side alert log. There is no server push channel, so entries are
/// derived from `/status` edges (risk level, breach, weapon) and the
/// `/api/snapshots` + `/api/anpr/results` feeds.
///
/// Alarm-grade entries ([AlertSeverity.critical]) stay unacknowledged until an
/// operator acknowledges them; the console keeps signalling until then.
class AlertLog extends ChangeNotifier {
  static const _cap = 300;

  final List<AlertEntry> _entries = [];
  List<AlertEntry> get entries => List.unmodifiable(_entries);

  final Map<int, String> _lastLevel = {};
  final Map<int, bool> _lastBreach = {};
  final Map<int, bool> _lastWeapon = {};
  final Map<int, bool> _lastAim = {};
  final Map<int, bool> _lastFace = {};
  final _seenPlates = _BoundedSet(600);
  final _seenSnaps = _BoundedSet(1200);
  final _seenAnpr = _BoundedSet(1200);
  final _seenFace = _BoundedSet(1200);

  // The first snapshot / ANPR / face poll returns records that already
  // existed when the console connected. They belong in the log as history,
  // but must not raise alarms for events the operator never missed.
  bool _snapsPrimed = false;
  bool _anprPrimed = false;
  bool _facePrimed = false;

  int _seq = 0;

  List<AlertEntry> get unacknowledged =>
      _entries.where((e) => e.needsAck).toList(growable: false);

  int get unackedCount => _entries.where((e) => e.needsAck).length;

  AlertEntry? get latestUnacked {
    for (final e in _entries) {
      if (e.needsAck) return e;
    }
    return null;
  }

  void acknowledge(AlertEntry e) {
    if (!e.needsAck) return;
    e.ackedAt = DateTime.now();
    notifyListeners();
  }

  void acknowledgeAll() {
    final now = DateTime.now();
    var changed = false;
    for (final e in _entries) {
      if (e.needsAck) {
        e.ackedAt = now;
        changed = true;
      }
    }
    if (changed) notifyListeners();
  }

  /// Forget everything, including de-duplication and priming state — used
  /// when the console is pointed at a different server.
  void clear() {
    _entries.clear();
    _lastLevel.clear();
    _lastBreach.clear();
    _lastWeapon.clear();
    _lastAim.clear();
    _lastFace.clear();
    _seenPlates.clear();
    _seenSnaps.clear();
    _seenAnpr.clear();
    _seenFace.clear();
    _snapsPrimed = false;
    _anprPrimed = false;
    _facePrimed = false;
    notifyListeners();
  }

  void ingestStatus(ServerStatus s) {
    var changed = false;
    final now = DateTime.now();

    for (final camId in s.cameraIds) {
      final m = s.metaFor(camId);
      final level = (m['level'] ?? 'Normal').toString();
      if (level != (_lastLevel[camId] ?? 'Normal')) {
        _lastLevel[camId] = level;
        if (level == 'High' || level == 'Critical') {
          _add(AlertEntry(
            id: ++_seq,
            kind: AlertKind.risk,
            severity: level == 'Critical'
                ? AlertSeverity.critical
                : AlertSeverity.warning,
            camId: camId,
            title: level.toUpperCase(),
            detail: 'score ${_num(m['score']).toStringAsFixed(0)} · '
                'P:${_num(m['persons']).toInt()} V:${_num(m['vehicles']).toInt()}',
            at: now,
          ));
          changed = true;
        }
      }

      final breach = m['breach'] == true;
      if (breach && !(_lastBreach[camId] ?? false)) {
        final zones = ((m['breach_zones'] as List?) ?? const []).join(', ');
        _add(AlertEntry(
          id: ++_seq,
          kind: AlertKind.breach,
          severity: AlertSeverity.critical,
          camId: camId,
          title: 'BREACH',
          detail: zones.isEmpty ? 'virtual fence' : zones,
          at: now,
        ));
        changed = true;
      }
      _lastBreach[camId] = breach;

      final weapon = m['weapon'] == true || m['armed'] == true;
      if (weapon && !(_lastWeapon[camId] ?? false)) {
        final tier = (m['weapon_tier'] ?? '').toString();
        _add(AlertEntry(
          id: ++_seq,
          kind: AlertKind.weapon,
          severity: AlertSeverity.critical,
          camId: camId,
          title: m['armed'] == true ? 'ARMED THREAT' : 'WEAPON',
          detail: tier.isEmpty ? 'firearm detected' : tier,
          at: now,
        ));
        changed = true;
      }
      _lastWeapon[camId] = weapon;

      // AIM skeleton heuristic without a confirmed gun: the server rates it
      // High, so it is a warning for the operator to look — not an alarm.
      final aim = !weapon && m['weapon_tier'] == 'posture';
      if (aim && !(_lastAim[camId] ?? false)) {
        _add(AlertEntry(
          id: ++_seq,
          kind: AlertKind.posture,
          severity: AlertSeverity.warning,
          camId: camId,
          title: 'AIM POSTURE',
          detail: 'two-handed aim stance · no weapon confirmed',
          at: now,
        ));
        changed = true;
      }
      _lastAim[camId] = aim;

      // Burst-confirmed watchlist face match — a named-identity signal, the
      // same override category as breach/weapon, never a generic "person
      // detected" alarm (see ibvap/risk_engine.py's criminal_match branch).
      final criminal = m['criminal_match'] == true;
      if (criminal && !(_lastFace[camId] ?? false)) {
        final name = (m['criminal_name'] ?? '').toString();
        _add(AlertEntry(
          id: ++_seq,
          kind: AlertKind.face,
          severity: AlertSeverity.critical,
          camId: camId,
          title: 'CRIMINAL SPOTTED',
          detail: name.isEmpty ? 'watchlist match' : name,
          at: now,
        ));
        changed = true;
      }
      _lastFace[camId] = criminal;
    }

    // Plates surfaced by ANPR (status.anpr.active[]).
    final active = (s.anpr['active'] as List?) ?? const [];
    for (final p in active.whereType<Map>()) {
      final plate = (p['plate'] ?? '').toString();
      final cam = (p['cam'] as num?)?.toInt() ?? 0;
      final conf = _num(p['conf']);
      if (plate.isEmpty || conf < 0.2) continue;
      if (_seenPlates.add('$cam|$plate')) {
        _add(AlertEntry(
          id: ++_seq,
          kind: AlertKind.plate,
          severity: AlertSeverity.info,
          camId: cam,
          title: 'PLATE',
          detail: '$plate  ${(conf * 100).toStringAsFixed(0)}%'
              '${p['valid'] == true ? '' : ' (unverified)'}',
          at: now,
        ));
        changed = true;
      }
    }

    if (changed) notifyListeners();
  }

  void ingestSnapshots(List<dynamic> rows) {
    final historical = !_snapsPrimed;
    _snapsPrimed = true;
    var changed = false;
    // oldest -> newest so the prepend order stays chronological
    for (final r in rows.reversed.whereType<Map>()) {
      final m = r.cast<String, dynamic>();
      final ts = _num(m['ts']);
      final reason = (m['reason'] ?? '').toString();
      final cam = (m['cam_id'] as num?)?.toInt() ?? 0;
      if (!_seenSnaps.add('$cam|$ts|$reason|${m['track_id']}')) continue;
      final kind = switch (reason) {
        'weapon' => AlertKind.weapon,
        'posture' => AlertKind.posture,
        'breach' => AlertKind.breach,
        _ => AlertKind.snapshot,
      };
      // Breach/weapon edges already alarm from /status; the snapshot adds the
      // evidence image, so it is logged without a second alarm.
      _add(AlertEntry(
        id: ++_seq,
        kind: kind,
        severity: kind == AlertKind.posture && !historical
            ? AlertSeverity.warning
            : AlertSeverity.info,
        camId: cam,
        title: '${reason.toUpperCase()} IMG',
        detail: (m['detail'] ?? '').toString(),
        at: DateTime.fromMillisecondsSinceEpoch((ts * 1000).round()),
        thumbFile: (m['file'] ?? m['raw_file'])?.toString(),
        historical: historical,
      ));
      changed = true;
    }
    if (changed) notifyListeners();
  }

  /// Event-triggered ANPR results (one row per vehicle arrival).
  void ingestAnprResults(List<dynamic> rows) {
    final historical = !_anprPrimed;
    _anprPrimed = true;
    var changed = false;
    for (final r in rows.reversed.whereType<Map>()) {
      final m = r.cast<String, dynamic>();
      final ts = _num(m['ts']);
      final cam = (m['cam_id'] as num?)?.toInt() ?? 0;
      if (!_seenAnpr.add('$cam|$ts|${m['track_id']}')) continue;
      final plate = (m['plate'] as String?);
      final vType = (m['vehicle_type'] as String?);
      final parts = <String>[
        if (plate?.isNotEmpty == true) plate!,
        if (vType?.isNotEmpty == true) vType!.replaceAll('_', ' '),
      ];
      _add(AlertEntry(
        id: ++_seq,
        kind: AlertKind.anpr,
        severity: AlertSeverity.info,
        camId: cam,
        title: 'VEHICLE',
        detail: parts.isEmpty ? 'plate unread' : parts.join(' · '),
        at: DateTime.fromMillisecondsSinceEpoch((ts * 1000).round()),
        thumbFile: (m['file'])?.toString(),
        vehicleType: vType,
        vehicleTypeConf: (m['vehicle_type_conf'] as num?)?.toDouble(),
        historical: historical,
      ));
      changed = true;
    }
    if (changed) notifyListeners();
  }

  /// Event-triggered face-recognition results (one row per person arrival's
  /// finalized burst — matched identity or "unknown"). The live Critical
  /// alarm already fired from ingestStatus's edge above, so this polled
  /// history feed never re-alarms — same convention ingestSnapshots uses.
  void ingestFaceResults(List<dynamic> rows) {
    final historical = !_facePrimed;
    _facePrimed = true;
    var changed = false;
    for (final r in rows.reversed.whereType<Map>()) {
      final m = r.cast<String, dynamic>();
      final ts = _num(m['ts']);
      final cam = (m['cam_id'] as num?)?.toInt() ?? 0;
      if (!_seenFace.add('$cam|$ts|${m['track_id']}')) continue;
      final onWatchlist = m['on_watchlist'] == true;
      final name = (m['matched_name'] as String?);
      final files = (m['files'] as List?) ?? const [];
      final votes = (m['votes'] as num?)?.toInt() ?? 0;
      final of = (m['of'] as num?)?.toInt() ?? 0;
      _add(AlertEntry(
        id: ++_seq,
        kind: AlertKind.face,
        severity: AlertSeverity.info,
        camId: cam,
        title: onWatchlist ? 'CRIMINAL SPOTTED (logged)' : 'FACE CHECKED',
        detail: onWatchlist
            ? (name?.isNotEmpty == true ? name! : 'watchlist match')
            : (of == 0 ? 'no face found' : '$votes/$of unmatched'),
        at: DateTime.fromMillisecondsSinceEpoch((ts * 1000).round()),
        thumbFile: files.isNotEmpty ? files.first.toString() : null,
        historical: historical,
      ));
      changed = true;
    }
    if (changed) notifyListeners();
  }

  void _add(AlertEntry e) {
    _entries.insert(0, e);
    if (_entries.length > _cap) {
      // Never silently drop an alarm the operator hasn't acknowledged.
      for (var i = _entries.length - 1; i >= 0 && _entries.length > _cap; i--) {
        if (!_entries[i].needsAck) _entries.removeAt(i);
      }
    }
  }

  static num _num(Object? v) => v is num ? v : 0;
}

/// Insertion-ordered set that evicts its oldest keys past [capacity].
///
/// The previous log cleared its whole de-dup set on overflow; the next poll
/// then re-inserted every record the server still returned, flooding the log
/// with duplicates of old events.
class _BoundedSet {
  _BoundedSet(this.capacity);
  final int capacity;
  final _keys = <String>{}; // set literals are insertion-ordered

  bool add(String k) {
    if (!_keys.add(k)) return false;
    if (_keys.length > capacity) _keys.remove(_keys.first);
    return true;
  }

  void clear() => _keys.clear();
}
