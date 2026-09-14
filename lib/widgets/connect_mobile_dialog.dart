import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:qr_flutter/qr_flutter.dart';

import '../state/app_state.dart';
import '../theme.dart';

class ConnectMobileDialog extends StatefulWidget {
  const ConnectMobileDialog({super.key, required this.state, this.initialCamId = 0});

  final AppState state;
  final int initialCamId;

  static void show(BuildContext context, AppState state, {int initialCamId = 0}) {
    showDialog(
      context: context,
      builder: (_) => ConnectMobileDialog(state: state, initialCamId: initialCamId),
    );
  }

  @override
  State<ConnectMobileDialog> createState() => _ConnectMobileDialogState();
}

class _ConnectMobileDialogState extends State<ConnectMobileDialog> {
  late int _selectedCamId = widget.initialCamId;
  Map<String, dynamic>? _info;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadInfo();
  }

  Future<void> _loadInfo() async {
    try {
      final res = await widget.state.client.mobileInfo();
      if (mounted) {
        setState(() {
          _info = res;
          _loading = false;
        });
      }
    } catch (_) {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    final lanIp = _info?['lan_ip'] ?? '127.0.0.1';
    final httpsPort = _info?['https_port'] ?? 8443;
    final haveCert = _info?['have_cert'] ?? true;
    final slots = (_info?['slots'] as List?)?.cast<Map<String, dynamic>>() ?? [];

    // A running Cloudflare tunnel gives a trusted certificate and works from
    // any network, so prefer it over the LAN address.
    final tunnelUrl = s.tunnel.phoneUrl(_selectedCamId);
    final activeUrl = tunnelUrl ?? 'https://$lanIp:$httpsPort/cam/$_selectedCamId';
    final devInfo = s.status?.devices.firstWhere(
      (d) => (d['cam_id'] as num?)?.toInt() == _selectedCamId,
      orElse: () => <String, dynamic>{},
    );
    final isConnected = devInfo?['connected'] == true;

    return Dialog(
      backgroundColor: const Color(0xFF0F172A),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: IbvapColors.border),
      ),
      insetPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
      child: Container(
        constraints: const BoxConstraints(maxWidth: 580),
        padding: const EdgeInsets.all(24),
        child: _loading
            ? const Center(
                child: Padding(
                  padding: EdgeInsets.all(40),
                  child: CircularProgressIndicator(color: IbvapColors.green),
                ),
              )
            : Column(
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
                        child: const Icon(Icons.phone_android_rounded, color: IbvapColors.green, size: 24),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Connect Smartphone Camera',
                              style: TextStyle(
                                color: IbvapColors.text,
                                fontSize: 16,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            Text(
                              'Stream live 720p/24fps video from any phone browser into the analytics grid.',
                              style: const TextStyle(color: IbvapColors.muted, fontSize: 11),
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

                  if (tunnelUrl != null) ...[
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: IbvapColors.tint(IbvapColors.green, 0.10),
                        border: Border.all(color: IbvapColors.green),
                        borderRadius: BorderRadius.circular(kRadius),
                      ),
                      child: const Row(children: [
                        Icon(Icons.public, color: IbvapColors.green, size: 16),
                        SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Using the Cloudflare tunnel — works from any network, '
                            'no certificate warning on the phone.',
                            style: TextStyle(color: IbvapColors.text, fontSize: 12),
                          ),
                        ),
                      ]),
                    ),
                    const SizedBox(height: 16),
                  ],
                  // Older servers don't report the flag; only warn on an explicit false.
                  if (tunnelUrl == null && _info?['lan_phone_intake'] == false) ...[
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: IbvapColors.tint(IbvapColors.orange, 0.12),
                        border: Border.all(color: IbvapColors.orange),
                        borderRadius: BorderRadius.circular(kRadius),
                      ),
                      child: const Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(Icons.lock_outline, color: IbvapColors.orange, size: 18),
                          SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              'Phone intake is disabled — the server only accepts '
                              'connections from this machine, so phones on the network '
                              'cannot reach these links. Set network.lan_phone_intake: '
                              'true in the backend config.yaml and restart the backend '
                              'to allow phone cameras for testing.',
                              style: TextStyle(color: IbvapColors.text, fontSize: 12),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                  ],

                  // Slot Picker Tabs
                  Row(
                    children: [
                      const Text(
                        'CAMERA SLOT:',
                        style: TextStyle(color: IbvapColors.muted, fontSize: 11, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(width: 12),
                      Wrap(
                        spacing: 8,
                        children: [
                          for (int i = 0; i < (slots.isEmpty ? 2 : slots.length); i++)
                            InkWell(
                              borderRadius: BorderRadius.circular(kRadius),
                              onTap: () => setState(() => _selectedCamId = i),
                              child: Container(
                                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                                decoration: BoxDecoration(
                                  color: _selectedCamId == i ? IbvapColors.green : const Color(0xFF1E293B),
                                  borderRadius: BorderRadius.circular(kRadius),
                                  border: Border.all(
                                    color: _selectedCamId == i ? IbvapColors.green : IbvapColors.border,
                                  ),
                                ),
                                child: Text(
                                  'CAM-$i',
                                  style: TextStyle(
                                    color: _selectedCamId == i ? Colors.black : IbvapColors.text,
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                            ),
                        ],
                      ),
                      const Spacer(),
                      // Connection Status
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: isConnected ? const Color(0x3322C55E) : const Color(0x33F59E0B),
                          borderRadius: BorderRadius.circular(kRadius),
                          border: Border.all(
                            color: isConnected ? IbvapColors.green : IbvapColors.orange,
                            width: 0.8,
                          ),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.circle,
                              size: 8,
                              color: isConnected ? IbvapColors.green : IbvapColors.orange,
                            ),
                            const SizedBox(width: 6),
                            Text(
                              isConnected ? 'STREAMING LIVE' : 'WAITING FOR SENDER',
                              style: TextStyle(
                                color: isConnected ? IbvapColors.green : IbvapColors.orange,
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),

                  // Center QR Code + URL
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // QR Code Card
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(kRadius),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0x4D000000),
                              blurRadius: 10,
                            ),
                          ],
                        ),
                        child: QrImageView(
                          data: activeUrl,
                          version: QrVersions.auto,
                          size: 160,
                          backgroundColor: Colors.white,
                        ),
                      ),
                      const SizedBox(width: 20),
                      // Step by step guide
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            _step(1, 'Connect phone to the same Wi-Fi network as this PC.'),
                            const SizedBox(height: 10),
                            _step(2, 'Scan the QR code with your phone camera app, or open:'),
                            const SizedBox(height: 6),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                              decoration: BoxDecoration(
                                color: const Color(0xFF020617),
                                borderRadius: BorderRadius.circular(kRadius),
                                border: Border.all(color: IbvapColors.border),
                              ),
                              child: Row(
                                children: [
                                  Expanded(
                                    child: SelectableText(
                                      activeUrl,
                                      style: const TextStyle(
                                        color: IbvapColors.green,
                                        fontFamily: 'monospace',
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ),
                                  IconButton(
                                    tooltip: 'Copy Link',
                                    icon: const Icon(Icons.copy, size: 16, color: IbvapColors.muted),
                                    onPressed: () {
                                      Clipboard.setData(ClipboardData(text: activeUrl));
                                      ScaffoldMessenger.of(context).showSnackBar(
                                        const SnackBar(content: Text('Camera URL copied to clipboard')),
                                      );
                                    },
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 10),
                            _step(3, 'Tap "Allow Camera Permission". Video streams instantly!'),
                            if (!haveCert) ...[
                              const SizedBox(height: 8),
                              const Text(
                                '⚠️ Note: Self-signed TLS certificate. On first load, tap Advanced → Proceed.',
                                style: TextStyle(color: IbvapColors.orange, fontSize: 10.5),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  const Divider(color: IbvapColors.border, height: 1),
                  const SizedBox(height: 14),

                  // Footer
                  Row(
                    children: [
                      const Icon(Icons.info_outline, size: 16, color: IbvapColors.muted),
                      const SizedBox(width: 8),
                      const Text(
                        'Adaptive JPEG over WebSocket with sub-100ms latency control.',
                        style: TextStyle(color: IbvapColors.muted, fontSize: 11),
                      ),
                      const Spacer(),
                      FilledButton(
                        style: FilledButton.styleFrom(
                          backgroundColor: const Color(0xFF334155),
                          foregroundColor: Colors.white,
                        ),
                        onPressed: () => Navigator.pop(context),
                        child: const Text('Done'),
                      ),
                    ],
                  ),
                ],
              ),
      ),
    );
  }

  Widget _step(int num, String text) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 20,
          height: 20,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: const Color(0x3322C55E),
            shape: BoxShape.circle,
            border: Border.all(color: IbvapColors.green, width: 0.8),
          ),
          child: Text(
            '$num',
            style: const TextStyle(
              color: IbvapColors.green,
              fontSize: 10,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            text,
            style: const TextStyle(color: IbvapColors.text, fontSize: 12),
          ),
        ),
      ],
    );
  }
}
