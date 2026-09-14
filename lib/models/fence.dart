/// A virtual fence, mirroring `ibvap/geofence.py` `Fence`.
///
/// Geometry is normalised 0..1 image coordinates for one camera:
///   * `polygon` — no-go zone, 3+ points (open ring)
///   * `line`    — tripwire polyline, 2+ points (each consecutive pair a segment)
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
  });

  final String? id;
  final int camId;
  final String kind; // 'polygon' | 'line'
  final List<(double, double)> points;
  final String direction; // 'both' | 'a2b' | 'b2a'  (line only)
  final List<String> targets; // ['any'] | ['person'] | ['vehicle']
  final String label;
  final bool enabled;

  bool get isLine => kind == 'line';

  factory Fence.fromJson(Map<String, dynamic> j) {
    final pts = <(double, double)>[];
    for (final p in (j['points'] as List? ?? const [])) {
      if (p is List && p.length >= 2) {
        pts.add(((p[0] as num).toDouble(), (p[1] as num).toDouble()));
      }
    }
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
    );
  }

  Map<String, dynamic> toJson() => {
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
