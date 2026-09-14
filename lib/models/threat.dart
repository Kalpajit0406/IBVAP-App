import 'package:flutter/material.dart';

import '../theme.dart';
import 'server_status.dart';

/// Force-wide threat condition shown in the command header.
///
/// [unknown] is a first-class state: with no live link or no reporting
/// cameras the console must never claim the area is nominal.
enum ThreatLevel {
  unknown('UNKNOWN', IbvapColors.muted),
  nominal('NOMINAL', IbvapColors.green),
  elevated('ELEVATED', IbvapColors.orange),
  critical('CRITICAL', IbvapColors.red);

  const ThreatLevel(this.label, this.color);
  final String label;
  final Color color;
}

class ThreatPicture {
  const ThreatPicture({
    required this.level,
    required this.reasons,
    required this.hotCams,
    required this.camsReporting,
  });

  final ThreatLevel level;

  /// Short operator-readable drivers, most severe first, e.g.
  /// `["WEAPON CAM-02", "BREACH CAM-00"]`.
  final List<String> reasons;

  /// Cameras at elevated or critical, most severe first.
  final List<int> hotCams;
  final int camsReporting;

  static const unknown = ThreatPicture(
      level: ThreatLevel.unknown, reasons: [], hotCams: [], camsReporting: 0);

  /// Derives the picture from one `/status` payload. [linkUp] false (stale or
  /// offline) always yields [ThreatLevel.unknown].
  factory ThreatPicture.from(ServerStatus? s, {required bool linkUp}) {
    if (s == null || !linkUp) return unknown;
    final ids = s.cameraIds;
    if (ids.isEmpty) return unknown;

    final crit = <String>[];
    final high = <String>[];
    final critCams = <int>[];
    final highCams = <int>[];

    for (final id in ids) {
      final m = s.metaFor(id);
      final cam = camLabel(id);
      final level = (m['level'] ?? 'Normal').toString();
      final weapon = m['weapon'] == true || m['armed'] == true;
      final breach = m['breach'] == true;
      final anomalies = (m['anomalies'] as List?)?.isNotEmpty ?? false;

      if (weapon) crit.add('WEAPON $cam');
      if (breach) crit.add('BREACH $cam');
      if (level == 'Critical' && !weapon && !breach) crit.add('CRITICAL $cam');
      if (weapon || breach || level == 'Critical') {
        critCams.add(id);
        continue;
      }
      if (level == 'High') high.add('HIGH $cam');
      if (anomalies) high.add('POSTURE $cam');
      if (level == 'High' || anomalies) highCams.add(id);
    }

    final level = crit.isNotEmpty
        ? ThreatLevel.critical
        : high.isNotEmpty
            ? ThreatLevel.elevated
            : ThreatLevel.nominal;
    return ThreatPicture(
      level: level,
      reasons: [...crit, ...high],
      hotCams: [...critCams, ...highCams],
      camsReporting: ids.length,
    );
  }
}

String camLabel(int id) => 'CAM-${id.toString().padLeft(2, '0')}';
