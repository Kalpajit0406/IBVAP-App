/// A virtual fence, mirroring `ibvap/geofence.py` `Fence`.
///
/// Geometry is normalised 0..1 image coordinates for one camera:
///   * `polygon` — no-go zone, 3+ points (open ring)
///   * `line`    — tripwire polyline, 2+ points (each consecutive pair a segment)
///
/// The console saves by read-modify-write of the WHOLE fence list
/// (`POST /api/fences` replaces the set), so anything this class fails to carry
/// through [fromJson] → [toJson] is silently erased from every fence on every
/// save. That is exactly how the per-fence behaviour fields were being wiped.
/// [extra] exists so that can never happen again: it holds every key this
/// client does not itself manage and writes it back untouched, which keeps
/// server-owned fields (`created_at`) and any field added to the backend later
/// intact even before this class learns about it.
class Fence {
  Fence({
    this.id,
    required this.camId,
    required this.kind,
    required this.points,
    this.direction = 'both',
    this.targets = const ['any'],
    this.label = '',
    this.enabled = true,
    this.severity = 'Critical',
    this.loiterAfterS = 0,
    this.armedFrom = '',
    this.armedTo = '',
    this.inbound = '',
    this.followWindowS = 0,
    this.extra = const {},
  });

  /// Alert severity choices, in the order the server ranks them.
  static const severities = ['Info', 'Low', 'Medium', 'High', 'Critical'];

  /// Keys this class reads and writes itself; everything else goes to [extra].
  static const _known = {
    'id', 'cam_id', 'kind', 'points', 'direction', 'targets', 'label',
    'enabled', 'severity', 'loiter_after_s', 'armed_from', 'armed_to',
    'inbound', 'follow_window_s',
  };

  final String? id;
  final int camId;
  final String kind; // 'polygon' | 'line'
  final List<(double, double)> points;
  final String direction; // 'both' | 'a2b' | 'b2a'  (line only)
  final List<String> targets; // ['any'] | ['person'] | ['vehicle']
  final String label;
  final bool enabled;

  /// How serious a breach of this fence is. A perimeter wire is Critical; a
  /// counting line across an approach road is not the same alarm.
  final String severity;

  /// Raise a separate loiter alert once a target has been inside a zone this
  /// long, in seconds. 0 = off. Zones only.
  final double loiterAfterS;

  /// Local-time arming window, 24-hour "HH:MM". Both empty = always armed.
  final String armedFrom;
  final String armedTo;

  /// Which line-crossing direction means "into our territory": '' (unset),
  /// 'a2b' or 'b2a'. There is no camera calibration, so this is whatever the
  /// person who drew the line says it is. Lines only.
  final String inbound;

  /// Lines only: raise a close-following alert when a second target of the
  /// same kind crosses the same way within this many seconds of another. 0 =
  /// off. It reports a pattern (two crossings close together), not a verdict —
  /// vision cannot tell whether the first crosser was authorised.
  final double followWindowS;

  /// Keys the console does not manage, preserved verbatim across a save.
  final Map<String, dynamic> extra;

  bool get isLine => kind == 'line';

  factory Fence.fromJson(Map<String, dynamic> j) {
    final pts = <(double, double)>[];
    for (final p in (j['points'] as List? ?? const [])) {
      if (p is List && p.length >= 2) {
        pts.add(((p[0] as num).toDouble(), (p[1] as num).toDouble()));
      }
    }
    final loiter = j['loiter_after_s'];
    final follow = j['follow_window_s'];
    return Fence(
      id: j['id']?.toString(),
      camId: (j['cam_id'] as num?)?.toInt() ?? 0,
      kind: (j['kind'] ?? 'polygon').toString(),
      points: pts,
      direction: (j['direction'] ?? 'both').toString(),
      targets: ((j['targets'] as List?) ?? const ['any'])
          .map((e) => e.toString())
          .toList(),
      label: (j['label'] ?? '').toString(),
      enabled: j['enabled'] != false,
      severity: (j['severity'] ?? 'Critical').toString(),
      loiterAfterS: loiter is num ? loiter.toDouble() : 0,
      armedFrom: (j['armed_from'] ?? '').toString(),
      armedTo: (j['armed_to'] ?? '').toString(),
      inbound: (j['inbound'] ?? '').toString(),
      followWindowS: follow is num ? follow.toDouble() : 0,
      extra: {
        for (final e in j.entries)
          if (!_known.contains(e.key)) e.key: e.value,
      },
    );
  }

  Map<String, dynamic> toJson() => {
        // Unknown keys first, so a key this class manages can never be
        // shadowed by a stale copy of itself sitting in [extra].
        ...extra,
        if (id != null) 'id': id,
        'cam_id': camId,
        'kind': kind,
        'points': [
          for (final (u, v) in points)
            [double.parse(u.toStringAsFixed(5)), double.parse(v.toStringAsFixed(5))]
        ],
        'direction': isLine ? direction : 'both',
        'targets': targets,
        'label': label,
        'enabled': enabled,
        'severity': severity,
        'loiter_after_s': loiterAfterS,
        'armed_from': armedFrom,
        'armed_to': armedTo,
        'inbound': inbound,
        'follow_window_s': followWindowS,
      };

  Fence copyWith({
    String? id,
    int? camId,
    String? kind,
    List<(double, double)>? points,
    String? direction,
    List<String>? targets,
    String? label,
    bool? enabled,
    String? severity,
    double? loiterAfterS,
    String? armedFrom,
    String? armedTo,
    String? inbound,
    double? followWindowS,
    Map<String, dynamic>? extra,
  }) =>
      Fence(
        id: id ?? this.id,
        camId: camId ?? this.camId,
        kind: kind ?? this.kind,
        points: points ?? this.points,
        direction: direction ?? this.direction,
        targets: targets ?? this.targets,
        label: label ?? this.label,
        enabled: enabled ?? this.enabled,
        severity: severity ?? this.severity,
        loiterAfterS: loiterAfterS ?? this.loiterAfterS,
        armedFrom: armedFrom ?? this.armedFrom,
        armedTo: armedTo ?? this.armedTo,
        inbound: inbound ?? this.inbound,
        followWindowS: followWindowS ?? this.followWindowS,
        extra: extra ?? this.extra,
      );
}

/// One tile of the composed mosaic, from `/status.mosaic.tiles`
/// (`{cam_id, x, y, w, h}` in canvas pixels).
class MosaicTile {
  MosaicTile(this.camId, this.x, this.y, this.w, this.h);

  final int camId;
  final double x, y, w, h;

  factory MosaicTile.fromJson(Map<String, dynamic> j) => MosaicTile(
        (j['cam_id'] as num).toInt(),
        (j['x'] as num).toDouble(),
        (j['y'] as num).toDouble(),
        (j['w'] as num).toDouble(),
        (j['h'] as num).toDouble(),
      );

  bool contains(double px, double py) =>
      px >= x && px < x + w && py >= y && py < y + h;
}

class MosaicLayout {
  MosaicLayout({
    required this.mode,
    required this.mains,
    required this.canvasW,
    required this.canvasH,
    required this.tiles,
    required this.ids,
  });

  final String mode; // 'grid' | 'focus'
  final List<int> mains;
  final double canvasW, canvasH;
  final List<MosaicTile> tiles;
  final List<int> ids;

  static MosaicLayout fromStatus(Map<String, dynamic> mosaic) {
    final tiles = ((mosaic['tiles'] as List?) ?? const [])
        .whereType<Map>()
        .map((e) => MosaicTile.fromJson(e.cast<String, dynamic>()))
        .toList();
    final canvas = (mosaic['canvas'] as List?)?.cast<num>();
    return MosaicLayout(
      mode: (mosaic['mode'] ?? 'grid').toString(),
      mains: ((mosaic['mains'] as List?) ?? const [])
          .map((e) => (e as num).toInt())
          .toList(),
      canvasW: (canvas != null && canvas.isNotEmpty) ? canvas[0].toDouble() : 1,
      canvasH: (canvas != null && canvas.length > 1) ? canvas[1].toDouble() : 1,
      tiles: tiles,
      ids: ((mosaic['ids'] as List?) ?? const [])
          .map((e) => (e as num).toInt())
          .toList(),
    );
  }

  MosaicTile? tileForCam(int camId) {
    for (final t in tiles) {
      if (t.camId == camId) return t;
    }
    return null;
  }

  MosaicTile? tileAtCanvas(double px, double py) {
    for (final t in tiles) {
      if (t.contains(px, py)) return t;
    }
    return null;
  }
}
