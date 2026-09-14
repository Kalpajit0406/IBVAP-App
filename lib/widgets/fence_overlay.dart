import 'package:flutter/material.dart';

import '../models/fence.dart';
import '../state/fence_editor.dart';
import '../theme.dart';

/// Draws the virtual fences (and the in-progress draft) on top of the mosaic,
/// and turns taps into fence points. Sits in a box whose aspect ratio matches
/// `layout.canvas`, so canvas→box is a single uniform scale.
class FenceOverlay extends StatelessWidget {
  const FenceOverlay({
    super.key,
    required this.layout,
    required this.editor,
    required this.activeByCam, // {cam_id(str): {fence_id: bool}}
  });

  final MosaicLayout layout;
  final FenceEditor editor;
  final Map<String, dynamic> activeByCam;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, c) {
        final scale = layout.canvasW <= 0
            ? 1.0
            : c.maxWidth / layout.canvasW;

        void handleTap(Offset local) {
          if (!editor.armed) return;
          final cx = local.dx / scale;
          final cy = local.dy / scale;
          final tile = layout.tileAtCanvas(cx, cy);
          if (tile == null) return;
          final u = (cx - tile.x) / tile.w;
          final v = (cy - tile.y) / tile.h;
          editor.addPoint(tile.camId, u, v);
        }

        return GestureDetector(
          behavior: HitTestBehavior.translucent,
          onTapDown: (d) => handleTap(d.localPosition),
          child: CustomPaint(
            painter: _FencePainter(
              layout: layout,
              scale: scale,
              fences: editor.fences,
              activeByCam: activeByCam,
              draftCam: editor.draftCam,
              draftKind: editor.kind,
              draft: editor.draft,
              armed: editor.armed,
              direction: editor.direction,
            ),
            size: Size.infinite,
          ),
        );
      },
    );
  }
}

class _FencePainter extends CustomPainter {
  _FencePainter({
    required this.layout,
    required this.scale,
    required this.fences,
    required this.activeByCam,
    required this.draftCam,
    required this.draftKind,
    required this.draft,
    required this.armed,
    required this.direction,
  });

  final MosaicLayout layout;
  final double scale;
  final List<Fence> fences;
  final Map<String, dynamic> activeByCam;
  final int? draftCam;
  final String draftKind;
  final List<(double, double)> draft;
  final bool armed;
  final String direction;

  Offset? _toBox(int camId, double u, double v) {
    final t = layout.tileForCam(camId);
    if (t == null) return null;
    return Offset((t.x + u * t.w) * scale, (t.y + v * t.h) * scale);
  }

  bool _hot(Fence f) {
    final z = activeByCam['${f.camId}'];
    return z is Map && f.id != null && z[f.id] == true;
  }

  @override
  void paint(Canvas canvas, Size size) {
    for (final f in fences) {
      if (!f.enabled) continue;
      final pts = <Offset>[];
      for (final (u, v) in f.points) {
        final o = _toBox(f.camId, u, v);
        if (o != null) pts.add(o);
      }
      if (pts.length < 2) continue;
      final hot = _hot(f);
      final col = hot ? IbvapColors.red : IbvapColors.green;

      if (!f.isLine && hot) {
        canvas.drawPath(
          _poly(pts, close: true),
          Paint()
            ..style = PaintingStyle.fill
            ..color = IbvapColors.red.withValues(alpha: 0.22),
        );
      }
      canvas.drawPath(
        _poly(pts, close: !f.isLine),
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = f.isLine ? 3 : 2.5
          ..strokeJoin = StrokeJoin.round
          ..color = col,
      );
      if (f.isLine && f.direction != 'both') {
        _drawArrows(canvas, pts, f.direction, col);
      }
      _label(canvas, pts.first, f.label.isEmpty ? (f.id ?? '') : f.label, col);
    }

    // draft
    if (armed && draft.isNotEmpty && draftCam != null) {
      final pts = <Offset>[];
      for (final (u, v) in draft) {
        final o = _toBox(draftCam!, u, v);
        if (o != null) pts.add(o);
      }
      if (pts.isNotEmpty) {
        final path = _poly(pts, close: draftKind != 'line');
        canvas.drawPath(
          path,
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = 2.5
            ..color = IbvapColors.orange,
        );
        for (final p in pts) {
          canvas.drawCircle(p, 4.5, Paint()..color = IbvapColors.orange);
        }
      }
    }
  }

  Path _poly(List<Offset> pts, {required bool close}) {
    final p = Path()..moveTo(pts.first.dx, pts.first.dy);
    for (var i = 1; i < pts.length; i++) {
      p.lineTo(pts[i].dx, pts[i].dy);
    }
    if (close) p.close();
    return p;
  }

  void _drawArrows(Canvas canvas, List<Offset> pts, String dir, Color col) {
    final sign = dir == 'a2b' ? 1.0 : -1.0;
    final paint = Paint()
      ..color = col
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    for (var i = 0; i < pts.length - 1; i++) {
      final a = pts[i], b = pts[i + 1];
      final mid = Offset((a.dx + b.dx) / 2, (a.dy + b.dy) / 2);
      final seg = b - a;
      final n = seg.distance == 0 ? 1.0 : seg.distance;
      final perp = Offset(seg.dy / n, -seg.dx / n) * sign;
      final tip = mid + perp * 16;
      canvas.drawLine(mid, tip, paint);
      final back = (a - b) / n * 6;
      canvas.drawLine(tip, tip + Offset(back.dx - perp.dx * 4, back.dy - perp.dy * 4), paint);
      canvas.drawLine(tip, tip + Offset(back.dx + perp.dx * 4, back.dy + perp.dy * 4), paint);
    }
  }

  void _label(Canvas canvas, Offset at, String text, Color col) {
    if (text.isEmpty) return;
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.w700,
          shadows: const [Shadow(color: Colors.black, blurRadius: 3)],
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, at + const Offset(4, -18));
  }

  @override
  bool shouldRepaint(covariant _FencePainter old) =>
      old.fences != fences ||
      old.draft.length != draft.length ||
      old.armed != armed ||
      old.scale != scale ||
      old.activeByCam != activeByCam;
}
