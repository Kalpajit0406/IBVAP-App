import 'package:flutter/material.dart';

import '../theme.dart';

enum AlertKind { risk, breach, plate, weapon, posture, snapshot, anpr }

/// How loudly an entry is signalled. Only [critical] raises the console alarm
/// and demands acknowledgement.
enum AlertSeverity { info, warning, critical }

/// One row in the live alert log. Derived client-side from `/status` edges,
/// `/api/snapshots`, and `/api/anpr/results`.
class AlertEntry {
  AlertEntry({
    required this.id,
    required this.kind,
    required this.camId,
    required this.title,
    required this.detail,
    required this.at,
    this.severity = AlertSeverity.info,
    this.thumbFile,
    this.vehicleType,
    this.vehicleTypeConf,
    this.historical = false,
  });

  final int id;
  final AlertKind kind;
  final AlertSeverity severity;
  final int camId;
  final String title; // e.g. "CRITICAL", "BREACH", "PLATE"
  final String detail;
  final DateTime at;
  final String? thumbFile; // "<cam>/<name>.jpg" — /snap/ for every kind except
                           // AlertKind.anpr, which serves from /anpr_snap/
  final String? vehicleType;     // AlertKind.anpr only
  final double? vehicleTypeConf; // AlertKind.anpr only

  /// Already on the server before this console connected — logged, not alarmed.
  final bool historical;

  DateTime? ackedAt;

  bool get needsAck =>
      severity == AlertSeverity.critical && !historical && ackedAt == null;

  String get cam => 'CAM-${camId.toString().padLeft(2, '0')}';

  Color get color => switch (kind) {
        AlertKind.breach => IbvapColors.red,
        AlertKind.weapon => IbvapColors.red,
        AlertKind.posture => IbvapColors.orange,
        AlertKind.plate => IbvapColors.plate,
        AlertKind.snapshot => IbvapColors.blue,
        AlertKind.anpr => IbvapColors.blue,
        AlertKind.risk => severity == AlertSeverity.critical
            ? IbvapColors.red
            : IbvapColors.orange,
      };

  IconData get icon => switch (kind) {
        AlertKind.breach => Icons.fence_outlined,
        AlertKind.weapon => Icons.gps_fixed,
        AlertKind.posture => Icons.accessibility_new,
        AlertKind.plate => Icons.pin_outlined,
        AlertKind.snapshot => Icons.photo_camera_outlined,
        AlertKind.anpr => Icons.local_shipping_outlined,
        AlertKind.risk => Icons.warning_amber_rounded,
      };
}
