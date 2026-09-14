import 'package:flutter/material.dart';

import '../../models/server_status.dart';
import '../../state/app_state.dart';
import '../../theme.dart';
import '../../widgets/backend_offline_card.dart';
import '../../widgets/model_switcher.dart';
import '../../widgets/panel.dart';
import '../../util/describe_error.dart';

class OverviewPanel extends StatelessWidget {
  const OverviewPanel({super.key, required this.state});

  final AppState state;

  Future<void> _reconnect(BuildContext context, int camId) async {
    try {
      await state.client.reconnectCamera(camId);
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('CAM-$camId reconnect requested')));
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Reconnect failed: ${describeError(e)}')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = state.status;
    if (s == null) {
      return ListView(
        children: [
          BackendOfflineCard(state: state),
        ],
      );
    }

    return ListView(
      children: [
        Panel(
          title: 'Model',
          child: ModelSwitcher(state: state),
        ),
        const SizedBox(height: 12),
        Panel(
          title: 'Pipeline',
          child: Wrap(
            spacing: 28,
            runSpacing: 18,
            children: [
              StatTile('device', s.onGpu ? 'CUDA' : s.device.toUpperCase(),
                  color: s.onGpu ? IbvapColors.green : IbvapColors.orange),
              StatTile('model', s.activeModel,
                  color: s.switching ? IbvapColors.orange : null),
              StatTile('fps in', s.fps.toStringAsFixed(0)),
              StatTile('batch', s.meanBatch.toStringAsFixed(1)),
              StatTile('ms / frame', s.inferenceMs.toStringAsFixed(1)),
              StatTile('gpu saved', '${s.gpuSavingPct.toStringAsFixed(0)}%',
                  color: IbvapColors.orange),
              StatTile('mux tps', s.muxerTps.toStringAsFixed(1)),
              StatTile('dropped', '${s.snapshotsDropped}'),
              StatTile('frames', '${s.framesProcessed}'),
              StatTile('input', s.inputMode),
            ],
          ),
        ),
        const SizedBox(height: 12),
        Panel(
          title: 'Cameras  (${s.camerasLive}/${s.cameraCount} live)',
          child: s.devices.isEmpty
              ? const Text('No cameras connected.',
                  style: TextStyle(color: IbvapColors.muted, fontSize: 12))
              : Column(
                  children: [
                    for (final d in s.devices)
                      _DeviceRow(
                        d,
                        breaching: s.camBreaching(
                            (d['cam_id'] as num?)?.toInt() ?? -1),
                        onReconnect: () => _reconnect(
                            context, (d['cam_id'] as num?)?.toInt() ?? 0),
                      ),
                  ],
                ),
        ),
        const SizedBox(height: 12),
        Panel(
          title: 'Subsystems',
          child: Wrap(
            spacing: 28,
            runSpacing: 18,
            children: [
              StatTile('fences', '${_fenceCount(s)}'),
              StatTile('snapshots', '${s.snapshots['written'] ?? 0}'),
              _flag('anpr', s.anpr['enabled'] == true),
              _flag('weapon', s.weapon['enabled'] == true),
              _flag('geofence', s.geofence['enabled'] == true),
              _flag('learning', s.learning['enabled'] == true),
            ],
          ),
        ),
      ],
    );
  }

  StatTile _flag(String label, bool on) => StatTile(
        label,
        on ? 'on' : 'off',
        color: on ? IbvapColors.green : IbvapColors.muted,
      );

  int _fenceCount(ServerStatus s) {
    final f = s.geofence['fences'];
    if (f is Map) {
      var n = 0;
      for (final v in f.values) {
        if (v is List) n += v.length;
      }
      return n;
    }
    return 0;
  }
}

class _DeviceRow extends StatelessWidget {
  const _DeviceRow(this.d, {required this.breaching, required this.onReconnect});
  final Map<String, dynamic> d;
  final bool breaching;
  final VoidCallback onReconnect;

  @override
  Widget build(BuildContext context) {
    final state = (d['state'] ?? 'offline').toString();
    final color = switch (state) {
      'live' => IbvapColors.green,
      'idle' => IbvapColors.orange,
      _ => IbvapColors.red,
    };
    final cam = 'CAM-${d['cam_id'].toString().padLeft(2, '0')}';
    final fps = (d['delivered_fps'] as num?)?.toStringAsFixed(0) ?? '0';
    final skip = (d['gate_skip_pct'] as num?)?.toStringAsFixed(0) ?? '0';
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
          const SizedBox(width: 8),
          Text(d['label']?.toString() ?? cam,
              style: const TextStyle(
                  color: IbvapColors.text,
                  fontSize: 12,
                  fontWeight: FontWeight.w600)),
          const SizedBox(width: 8),
          Text(cam,
              style: const TextStyle(color: IbvapColors.muted, fontSize: 11)),
          if (breaching) ...[
            const SizedBox(width: 8),
            const Text('BREACH',
                style: TextStyle(
                    color: IbvapColors.red,
                    fontSize: 9,
                    fontWeight: FontWeight.w700)),
          ],
          const Spacer(),
          Text('$fps fps · gate $skip%',
              style: const TextStyle(color: IbvapColors.muted, fontSize: 11)),
          const SizedBox(width: 6),
          Text(state.toUpperCase(),
              style: TextStyle(
                  color: color, fontSize: 9, fontWeight: FontWeight.w700)),
          IconButton(
            icon: const Icon(Icons.refresh, size: 15),
            color: IbvapColors.muted,
            visualDensity: VisualDensity.compact,
            tooltip: 'Reconnect (RTSP)',
            onPressed: onReconnect,
          ),
        ],
      ),
    );
  }
}
