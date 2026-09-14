import 'package:flutter/material.dart';

import '../../state/app_state.dart';
import '../../theme.dart';
import '../../widgets/add_camera_dialog.dart';
import '../../widgets/backend_offline_card.dart';
import '../../widgets/connect_mobile_dialog.dart';
import '../../widgets/panel.dart';
import '../../util/describe_error.dart';

class StreamsPanel extends StatefulWidget {
  const StreamsPanel({super.key, required this.state});

  final AppState state;

  @override
  State<StreamsPanel> createState() => _StreamsPanelState();
}

class _StreamsPanelState extends State<StreamsPanel> {
  List<Map<String, dynamic>> _streams = [];
  bool _loading = false;
  String? _lanIp;

  @override
  void initState() {
    super.initState();
    _loadStreams();
  }

  Future<void> _loadStreams() async {
    if (_loading) return;
    setState(() => _loading = true);
    try {
      final res = await widget.state.client.streams();
      if (mounted) {
        setState(() {
          _lanIp = res['lan_ip']?.toString();
          _streams = (res['streams'] as List?)?.cast<Map<String, dynamic>>() ?? [];
          _loading = false;
        });
      }
    } catch (_) {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _deleteStream(int camId, String name) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: const Color(0xFF1E293B),
        title: Text('Delete Camera $name?'),
        content: Text('This will remove CAM-$camId from the surveillance grid and configuration.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: IbvapColors.red),
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirm != true) return;

    try {
      await widget.state.client.deleteStream(camId);
      await _loadStreams();
      await widget.state.refreshNow();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Camera $name deleted.')));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error deleting stream: ${describeError(e)}')));
      }
    }
  }

  Future<void> _reconnect(int camId) async {
    try {
      await widget.state.client.reconnectCamera(camId);
      await widget.state.refreshNow();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('CAM-$camId reconnect requested.')));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Reconnect failed: ${describeError(e)}')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    if (s.link == LinkState.offline) {
      return ListView(
        padding: const EdgeInsets.all(16),
        children: [
          BackendOfflineCard(state: s),
        ],
      );
    }

    final liveDevices = s.status?.devices ?? [];

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // ── Action Bar Header ──────────────────────────────────────────────
        Panel(
          title: 'Camera Streams & Ingestion Management',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  FilledButton.icon(
                    style: FilledButton.styleFrom(
                      backgroundColor: IbvapColors.green,
                      foregroundColor: Colors.black,
                    ),
                    icon: const Icon(Icons.add_link_rounded, size: 18),
                    label: const Text('Add CCTV / RTSP Camera'),
                    onPressed: () async {
                      AddCameraDialog.show(context, s);
                      await Future.delayed(const Duration(milliseconds: 500));
                      _loadStreams();
                    },
                  ),
                  const SizedBox(width: 10),
                  FilledButton.icon(
                    style: FilledButton.styleFrom(
                      backgroundColor: const Color(0xFF38BDF8),
                      foregroundColor: Colors.black,
                    ),
                    icon: const Icon(Icons.phone_android_rounded, size: 18),
                    label: const Text('Connect Phone (QR Code)'),
                    onPressed: () => ConnectMobileDialog.show(context, s),
                  ),
                  const Spacer(),
                  IconButton(
                    tooltip: 'Refresh Camera List',
                    icon: const Icon(Icons.refresh_rounded, color: IbvapColors.muted),
                    onPressed: _loadStreams,
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Text(
                'Local Network IP: ${_lanIp ?? "127.0.0.1"} • Mobile Intake: https://${_lanIp ?? "127.0.0.1"}:8443/cam/N',
                style: const TextStyle(color: IbvapColors.muted, fontSize: 11),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),

        // ── Configured Streams List ────────────────────────────────────────
        Panel(
          title: 'Configured Surveillance Cameras (${_streams.length})',
          child: _loading
              ? const Center(
                  child: Padding(
                    padding: EdgeInsets.all(24),
                    child: CircularProgressIndicator(color: IbvapColors.green),
                  ),
                )
              : _streams.isEmpty
                  ? const Padding(
                      padding: EdgeInsets.all(16),
                      child: Text('No camera streams configured yet. Click "Add CCTV / RTSP Camera" or "Connect Phone".',
                          style: TextStyle(color: IbvapColors.muted, fontSize: 12)),
                    )
                  : Column(
                      children: [
                        for (final stream in _streams) _buildStreamCard(stream, liveDevices),
                      ],
                    ),
        ),
        const SizedBox(height: 16),

        // ── CCTV Presets & Guides ──────────────────────────────────────────
        Panel(
          title: 'Real CCTV & Mobile Ingestion Guide',
          child: DefaultTextStyle(
            style: const TextStyle(color: IbvapColors.muted, fontSize: 11.5, height: 1.4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  'Connecting Real CCTV / NVR Channels:',
                  style: TextStyle(color: IbvapColors.text, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 4),
                Text('• Hikvision Sub-stream:  rtsp://admin:password@IP:554/Streaming/Channels/102 (ch1 sub)'),
                Text('• Dahua / CP Plus Sub-stream:  rtsp://admin:password@IP:554/cam/realmonitor?channel=1&subtype=1'),
                Text('• Axis Sub-stream:  rtsp://admin:password@IP/axis-media/media.amp?resolution=640x360'),
                Text('• Uniview Sub-stream:  rtsp://admin:password@IP:554/media/video2'),
                Text('• Android IP Webcam App:  http://IP:8080/video'),
                SizedBox(height: 8),
                Text(
                  'Tip: Always use the low-resolution sub-stream (640x360 or 1280x720) for analytics to maximize GPU FPS throughput and save CPU decode overhead.',
                  style: TextStyle(color: IbvapColors.orange, fontSize: 11),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildStreamCard(
      Map<String, dynamic> stream, List<Map<String, dynamic>> liveDevices) {
    final camId = stream['id'] as int? ?? 0;
    final name = stream['name']?.toString() ?? 'CAM-$camId';
    final url = stream['url']?.toString() ?? 'ws';
    final type = stream['type']?.toString().toUpperCase() ?? 'RTSP';
    final isLive = stream['live'] == true;
    final isEnabled = stream['enabled'] == true;
    final isWs = type == 'PHONE' || url == 'ws';

    // NB: an explicit <String, dynamic>{} is required here — `liveDevices` is
    // reified as List<Map<String, dynamic>> at runtime, so a bare `{}` (which
    // infers as Map<dynamic, dynamic>) mismatches firstWhere's orElse type and
    // throws at runtime even though `flutter analyze` sees nothing wrong.
    final dev = liveDevices.firstWhere(
      (d) => (d['cam_id'] as num?)?.toInt() == camId,
      orElse: () => <String, dynamic>{},
    );
    final latency = (dev['latency_ms'] as num?)?.toInt();
    final frames = stream['frames_received'] ?? dev['frames_received'] ?? 0;

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: isLive ? const Color(0x1F22C55E) : const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(kRadius),
        border: Border.all(
          color: isLive ? IbvapColors.green : IbvapColors.border,
          width: isLive ? 1.2 : 0.8,
        ),
      ),
      child: Row(
        children: [
          // Icon & Type
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFF1E293B),
              borderRadius: BorderRadius.circular(kRadius),
            ),
            child: Icon(
              isWs
                  ? Icons.phone_android_rounded
                  : type == 'SCREEN'
                      ? Icons.desktop_windows_rounded
                      : type == 'WEBCAM'
                          ? Icons.camera_alt_rounded
                          : Icons.videocam_rounded,
              color: isLive ? IbvapColors.green : IbvapColors.muted,
              size: 22,
            ),
          ),
          const SizedBox(width: 14),

          // Details
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      name,
                      style: TextStyle(
                        color: isLive ? IbvapColors.green : IbvapColors.text,
                        fontWeight: FontWeight.bold,
                        fontSize: 13,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                      decoration: BoxDecoration(
                        color: const Color(0xFF1E293B),
                        borderRadius: BorderRadius.circular(4),
                        border: Border.all(color: IbvapColors.border),
                      ),
                      child: Text(
                        type,
                        style: const TextStyle(color: IbvapColors.muted, fontSize: 9, fontWeight: FontWeight.bold),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                      decoration: BoxDecoration(
                        color: isLive
                            ? const Color(0x3322C55E)
                            : isEnabled
                                ? const Color(0x33F59E0B)
                                : const Color(0x33EF4444),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        isLive
                            ? 'LIVE STREAMING'
                            : isEnabled
                                ? 'WAITING / IDLE'
                                : 'DISABLED',
                        style: TextStyle(
                          color: isLive
                              ? IbvapColors.green
                              : isEnabled
                                  ? IbvapColors.orange
                                  : IbvapColors.red,
                          fontSize: 9,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 3),
                Text(
                  isWs
                      ? 'Mobile WebSocket Slot: https://${_lanIp ?? "127.0.0.1"}:8443/cam/$camId'
                      : 'URL: $url  •  Transport: ${stream["transport"] ?? "tcp"}  •  Decode: ${stream["decode_fps"] ?? 15} FPS',
                  style: const TextStyle(color: IbvapColors.muted, fontSize: 11, fontFamily: 'monospace'),
                ),
                if (isLive) ...[
                  const SizedBox(height: 2),
                  Text(
                    'Frames Ingested: $frames ${latency != null ? " • Latency: ${latency}ms" : ""}',
                    style: const TextStyle(color: IbvapColors.green, fontSize: 10.5),
                  ),
                ],
              ],
            ),
          ),

          // Actions
          if (isWs)
            OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                side: const BorderSide(color: Color(0xFF38BDF8)),
                visualDensity: VisualDensity.compact,
              ),
              icon: const Icon(Icons.qr_code_2_rounded, size: 14, color: Color(0xFF38BDF8)),
              label: const Text('QR Code', style: TextStyle(color: Color(0xFF38BDF8), fontSize: 11)),
              onPressed: () => ConnectMobileDialog.show(context, widget.state, initialCamId: camId),
            )
          else ...[
            IconButton(
              tooltip: 'Reconnect Feed',
              icon: const Icon(Icons.refresh_rounded, size: 18, color: IbvapColors.muted),
              onPressed: () => _reconnect(camId),
            ),
            IconButton(
              tooltip: 'Edit Camera',
              icon: const Icon(Icons.edit_outlined, size: 18, color: IbvapColors.muted),
              onPressed: () async {
                AddCameraDialog.show(context, widget.state, editingStream: stream);
                await Future.delayed(const Duration(milliseconds: 500));
                _loadStreams();
              },
            ),
          ],
          IconButton(
            tooltip: 'Delete Camera',
            icon: const Icon(Icons.delete_outline, size: 18, color: IbvapColors.red),
            onPressed: () => _deleteStream(camId, name),
          ),
        ],
      ),
    );
  }
}
