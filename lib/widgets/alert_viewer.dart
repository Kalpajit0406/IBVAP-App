import 'package:flutter/material.dart';

import '../models/alert.dart';
import '../state/app_state.dart';
import '../theme.dart';

class AlertViewer {
  /// AlertKind.anpr thumbnails come from data/anpr_results/ (/anpr_snap/),
  /// AlertKind.face from data/face_results/ (/face_snap/), every other kind
  /// from data/snapshots/ (/snap/).
  static Uri thumbUrl(AppState state, AlertEntry e) => switch (e.kind) {
        AlertKind.anpr => state.client.anprSnapUrl(e.thumbFile!),
        AlertKind.face => state.client.faceSnapUrl(e.thumbFile!),
        _ => state.client.snapshotUrl(e.thumbFile!),
      };

  static Future<void> show(BuildContext context, AppState state, AlertEntry e) {
    return showDialog<void>(
      context: context,
      builder: (ctx) => Dialog(
        backgroundColor: IbvapColors.bg,
        insetPadding: const EdgeInsets.all(28),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                children: [
                  Container(width: 3, height: 18, color: e.color),
                  const SizedBox(width: 8),
                  Text(e.cam, style: IbvapText.data(size: 13, weight: FontWeight.w700)),
                  const SizedBox(width: 10),
                  Text(e.title, style: IbvapText.label(size: 12, color: e.color)),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(e.detail,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(color: IbvapColors.muted, fontSize: 12)),
                  ),
                  Text(_stamp(e.at), style: IbvapText.data(size: 11, color: IbvapColors.muted)),
                  if (e.needsAck) ...[
                    const SizedBox(width: 10),
                    FilledButton(
                      style: FilledButton.styleFrom(backgroundColor: IbvapColors.red,
                          foregroundColor: Colors.white),
                      onPressed: () {
                        state.alerts.acknowledge(e);
                        Navigator.pop(ctx);
                      },
                      child: const Text('ACKNOWLEDGE'),
                    ),
                  ],
                  IconButton(
                    onPressed: () => Navigator.pop(ctx),
                    icon: const Icon(Icons.close, size: 18),
                    color: IbvapColors.muted,
                  ),
                ],
              ),
              const SizedBox(height: 8),
              if (e.thumbFile != null)
                Flexible(
                  child: InteractiveViewer(
                    maxScale: 6,
                    child: Image.network(
                      thumbUrl(state, e).toString(),
                      gaplessPlayback: true,
                      errorBuilder: (_, _, _) => const SizedBox(
                        height: 200,
                        child: Center(
                          child: Text('Image unavailable',
                              style: TextStyle(color: IbvapColors.muted)),
                        ),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  static String _stamp(DateTime d) {
    String two(int v) => v.toString().padLeft(2, '0');
    return '${d.year}-${two(d.month)}-${two(d.day)} ${two(d.hour)}:${two(d.minute)}:${two(d.second)}';
  }
}
