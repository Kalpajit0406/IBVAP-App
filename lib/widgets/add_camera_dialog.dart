import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../theme.dart';
import '../util/describe_error.dart';

class AddCameraDialog extends StatefulWidget {
  const AddCameraDialog({super.key, required this.state, this.editingStream});

  final AppState state;
  final Map<String, dynamic>? editingStream;

  static void show(BuildContext context, AppState state, {Map<String, dynamic>? editingStream}) {
    showDialog(
      context: context,
      builder: (_) => AddCameraDialog(state: state, editingStream: editingStream),
    );
  }

  @override
  State<AddCameraDialog> createState() => _AddCameraDialogState();
}

class _AddCameraDialogState extends State<AddCameraDialog> {
  late final TextEditingController _nameCtrl;
  late final TextEditingController _urlCtrl;
  late int _camId;
  late String _transport = 'tcp';
  late double _zoneSensitivity = 0.7;
  late double _decodeFps = 15;
  bool _enabled = true;

  bool _probing = false;
  Map<String, dynamic>? _probeResult;
  bool _discovering = false;
  List<Map<String, dynamic>> _discoveredDevices = [];

  static const _presets = [
    (label: 'Custom RTSP / Camera', template: 'rtsp://admin:password@192.168.1.50:554/stream'),
    (label: 'Hikvision (Sub-stream ch1)', template: 'rtsp://admin:password@192.168.1.50:554/Streaming/Channels/102'),
    (label: 'Dahua / CP Plus (Sub-stream)', template: 'rtsp://admin:password@192.168.1.50:554/cam/realmonitor?channel=1&subtype=1'),
    (label: 'Axis (Sub-stream)', template: 'rtsp://admin:password@192.168.1.50/axis-media/media.amp?resolution=640x360'),
    (label: 'Uniview (Sub-stream)', template: 'rtsp://admin:password@192.168.1.50:554/media/video2'),
    (label: 'Android IP Webcam App', template: 'http://192.168.1.50:8080/video'),
    (label: 'USB Webcam 0', template: '0'),
    (label: 'USB Webcam 1', template: '1'),
    (label: 'Screen Capture', template: 'screen'),
    (label: 'Mobile Phone Slot', template: 'ws'),
  ];

  @override
  void initState() {
    super.initState();
    final edit = widget.editingStream;
    _camId = edit?['id'] as int? ?? _nextAvailableId();
    _nameCtrl = TextEditingController(text: edit?['name'] as String? ?? 'CAM-$_camId');
    final rawUrl = edit?['raw_url'] as String? ?? edit?['url'] as String? ?? 'rtsp://admin:password@192.168.1.50:554/Streaming/Channels/102';
    _urlCtrl = TextEditingController(text: rawUrl);
    _transport = edit?['transport'] as String? ?? 'tcp';
    _zoneSensitivity = (edit?['zone_sensitivity'] as num?)?.toDouble() ?? 0.7;
    _decodeFps = (edit?['decode_fps'] as num?)?.toDouble() ?? 15.0;
    _enabled = edit?['enabled'] as bool? ?? true;
  }

  int _nextAvailableId() {
    final existing = widget.state.status?.devices.map((d) => (d['cam_id'] as num?)?.toInt() ?? 0).toSet() ?? {};
    int id = 0;
    while (existing.contains(id)) {
      id++;
    }
    return id;
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _urlCtrl.dispose();
    super.dispose();
  }

  Future<void> _probe() async {
    final url = _urlCtrl.text.trim();
    if (url.isEmpty) return;
    setState(() {
      _probing = true;
      _probeResult = null;
    });
    try {
      final res = await widget.state.client.probeStream(url, transport: _transport);
      if (mounted) {
        setState(() {
          _probeResult = res;
          _probing = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _probeResult = {'ok': false, 'error': describeError(e)};
          _probing = false;
        });
      }
    }
  }

  Future<void> _discoverOnvif() async {
    setState(() {
      _discovering = true;
      _discoveredDevices.clear();
    });
    try {
      final res = await widget.state.client.discoverLanCameras();
      final devs = (res['devices'] as List?)?.cast<Map<String, dynamic>>() ?? [];
      if (mounted) {
        setState(() {
          _discoveredDevices = devs;
          _discovering = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _discovering = false);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Discovery error: ${describeError(e)}')));
      }
    }
  }

  Future<void> _save() async {
    final name = _nameCtrl.text.trim();
    final url = _urlCtrl.text.trim();
    if (name.isEmpty || url.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Name and URL are required.')),
      );
      return;
    }

    try {
      await widget.state.client.upsertStream({
        'id': _camId,
        'name': name,
        'url': url,
        'zone_sensitivity': _zoneSensitivity,
        'transport': _transport,
        'decode_fps': _decodeFps,
        'enabled': _enabled,
      });
      await widget.state.refreshNow();
      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Camera $name saved and loaded!')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to save camera: ${describeError(e)}')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: const Color(0xFF0F172A),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: IbvapColors.border),
      ),
      insetPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
      child: Container(
        constraints: const BoxConstraints(maxWidth: 620),
        padding: const EdgeInsets.all(24),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Header
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: const Color(0x3322C55E),
                      borderRadius: BorderRadius.circular(kRadius),
                    ),
                    child: const Icon(Icons.videocam_rounded, color: IbvapColors.green, size: 24),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          widget.editingStream == null ? 'Add Camera Stream' : 'Edit Camera CAM-$_camId',
                          style: const TextStyle(
                            color: IbvapColors.text,
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const Text(
                          'Configure RTSP CCTV, NVR channels, IP cameras, mobile intake or webcams.',
                          style: TextStyle(color: IbvapColors.muted, fontSize: 11),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    tooltip: 'Close',
                    icon: const Icon(Icons.close, color: IbvapColors.muted),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              const Divider(color: IbvapColors.border, height: 1),
              const SizedBox(height: 16),

              // Camera ID & Name Row
              Row(
                children: [
                  SizedBox(
                    width: 90,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('CAM ID', style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
                        const SizedBox(height: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 12),
                          decoration: BoxDecoration(
                            color: const Color(0xFF1E293B),
                            borderRadius: BorderRadius.circular(kRadius),
                            border: Border.all(color: IbvapColors.border),
                          ),
                          child: Text(
                            'CAM-$_camId',
                            style: const TextStyle(
                              color: IbvapColors.green,
                              fontWeight: FontWeight.bold,
                              fontSize: 13,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Camera Label / Name', style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
                        const SizedBox(height: 6),
                        TextField(
                          controller: _nameCtrl,
                          style: const TextStyle(color: IbvapColors.text, fontSize: 13),
                          decoration: const InputDecoration(hintText: 'e.g. North Gate / Perimeter / Phone-0'),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),

              // Vendor Presets Dropdown
              const Text('Preset / Source Template', style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
              const SizedBox(height: 6),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                decoration: BoxDecoration(
                  color: const Color(0xFF1E293B),
                  borderRadius: BorderRadius.circular(kRadius),
                  border: Border.all(color: IbvapColors.border),
                ),
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    isExpanded: true,
                    dropdownColor: const Color(0xFF1E293B),
                    hint: const Text('Choose a Vendor or Stream Preset…', style: TextStyle(color: IbvapColors.muted, fontSize: 12)),
                    items: _presets.map((p) {
                      return DropdownMenuItem(
                        value: p.template,
                        child: Text(p.label, style: const TextStyle(color: IbvapColors.text, fontSize: 12)),
                      );
                    }).toList(),
                    onChanged: (val) {
                      if (val != null) {
                        setState(() => _urlCtrl.text = val);
                      }
                    },
                  ),
                ),
              ),
              const SizedBox(height: 14),

              // Stream URL + Probe Button
              const Text('Stream URL (RTSP / HTTP / ws / 0 / screen / file)', style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
              const SizedBox(height: 6),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _urlCtrl,
                      style: const TextStyle(color: IbvapColors.text, fontSize: 13, fontFamily: 'monospace'),
                      decoration: const InputDecoration(hintText: 'rtsp://user:pass@ip:554/... or ws'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  FilledButton.icon(
                    style: FilledButton.styleFrom(
                      backgroundColor: const Color(0xFF334155),
                      foregroundColor: Colors.white,
                    ),
                    icon: _probing
                        ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                        : const Icon(Icons.network_check_rounded, size: 16),
                    label: const Text('Test'),
                    onPressed: _probing ? null : _probe,
                  ),
                ],
              ),

              // Probe Result Banner
              if (_probeResult != null) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: _probeResult!['ok'] == true ? const Color(0x2222C55E) : const Color(0x22EF4444),
                    borderRadius: BorderRadius.circular(kRadius),
                    border: Border.all(
                      color: _probeResult!['ok'] == true ? IbvapColors.green : IbvapColors.red,
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        _probeResult!['ok'] == true ? Icons.check_circle_rounded : Icons.error_rounded,
                        color: _probeResult!['ok'] == true ? IbvapColors.green : IbvapColors.red,
                        size: 18,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _probeResult!['ok'] == true
                              ? 'Stream verified! Resolution: ${_probeResult!["width"]}x${_probeResult!["height"]} • ${_probeResult!["fps"]} FPS • ${_probeResult!["latency_ms"]} ms'
                              : 'Connection failed: ${_probeResult!["error"]}',
                          style: TextStyle(
                            color: _probeResult!['ok'] == true ? IbvapColors.green : IbvapColors.red,
                            fontSize: 11.5,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              const SizedBox(height: 14),

              // ONVIF Discovery Trigger
              Row(
                children: [
                  OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: IbvapColors.border),
                      visualDensity: VisualDensity.compact,
                    ),
                    icon: _discovering
                        ? const SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 1.5, color: IbvapColors.green))
                        : const Icon(Icons.radar_rounded, size: 14, color: IbvapColors.green),
                    label: Text(
                      _discovering ? 'Scanning LAN…' : 'Scan LAN for ONVIF CCTV Cameras',
                      style: const TextStyle(color: IbvapColors.text, fontSize: 11),
                    ),
                    onPressed: _discovering ? null : _discoverOnvif,
                  ),
                ],
              ),

              // Discovered Cameras List
              if (_discoveredDevices.isNotEmpty) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: const Color(0xFF1E293B),
                    borderRadius: BorderRadius.circular(kRadius),
                    border: Border.all(color: IbvapColors.border),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Found ${_discoveredDevices.length} ONVIF device(s):',
                          style: const TextStyle(color: IbvapColors.muted, fontSize: 10, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      for (final dev in _discoveredDevices)
                        ListTile(
                          dense: true,
                          contentPadding: EdgeInsets.zero,
                          leading: const Icon(Icons.camera_alt, color: IbvapColors.green, size: 16),
                          title: Text('${dev["name"]} (${dev["ip"]})',
                              style: const TextStyle(color: IbvapColors.text, fontSize: 11.5)),
                          subtitle: Text('XAddr: ${dev["xaddr"]}',
                              style: const TextStyle(color: IbvapColors.muted, fontSize: 10)),
                          trailing: FilledButton(
                            style: FilledButton.styleFrom(visualDensity: VisualDensity.compact),
                            child: const Text('Use IP', style: TextStyle(fontSize: 10)),
                            onPressed: () {
                              setState(() {
                                _urlCtrl.text = 'rtsp://admin:password@${dev["ip"]}:554/Streaming/Channels/102';
                              });
                            },
                          ),
                        ),
                    ],
                  ),
                ),
              ],
              const SizedBox(height: 14),

              // Transport & Decode FPS & Zone Sensitivity
              Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Transport Protocol', style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
                        const SizedBox(height: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          decoration: BoxDecoration(
                            color: const Color(0xFF1E293B),
                            borderRadius: BorderRadius.circular(kRadius),
                            border: Border.all(color: IbvapColors.border),
                          ),
                          child: DropdownButtonHideUnderline(
                            child: DropdownButton<String>(
                              isExpanded: true,
                              value: _transport,
                              dropdownColor: const Color(0xFF1E293B),
                              items: const [
                                DropdownMenuItem(value: 'tcp', child: Text('TCP (Reliable, No Tears)', style: TextStyle(fontSize: 12))),
                                DropdownMenuItem(value: 'udp', child: Text('UDP (Low Latency)', style: TextStyle(fontSize: 12))),
                              ],
                              onChanged: (v) => setState(() => _transport = v ?? 'tcp'),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Zone Sensitivity: ${(_zoneSensitivity * 100).toInt()}%',
                            style: const TextStyle(color: IbvapColors.muted, fontSize: 11)),
                        Slider(
                          value: _zoneSensitivity,
                          min: 0.1,
                          max: 1.0,
                          divisions: 9,
                          activeColor: IbvapColors.green,
                          onChanged: (v) => setState(() => _zoneSensitivity = v),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              const Divider(color: IbvapColors.border, height: 1),
              const SizedBox(height: 16),

              // Action Buttons
              Row(
                children: [
                  Row(
                    children: [
                      Checkbox(
                        value: _enabled,
                        activeColor: IbvapColors.green,
                        checkColor: Colors.black,
                        onChanged: (v) => setState(() => _enabled = v ?? true),
                      ),
                      const Text('Enable Stream', style: TextStyle(color: IbvapColors.text, fontSize: 12)),
                    ],
                  ),
                  const Spacer(),
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Cancel'),
                  ),
                  const SizedBox(width: 8),
                  FilledButton.icon(
                    style: FilledButton.styleFrom(
                      backgroundColor: IbvapColors.green,
                      foregroundColor: Colors.black,
                    ),
                    icon: const Icon(Icons.check_rounded, size: 16),
                    label: const Text('Save & Load Camera'),
                    onPressed: _save,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
