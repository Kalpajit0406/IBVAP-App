import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:qr_flutter/qr_flutter.dart';

import '../config/app_config.dart';
import '../services/tunnel_manager.dart';
import '../state/app_state.dart';
import '../theme.dart';
import 'confirm_dialog.dart';
import 'panel.dart';

/// Settings section: publish the local product through a Cloudflare Tunnel so
/// phones (or a remote viewer) can reach it over valid HTTPS, without port
/// forwarding, firewall rules or certificate warnings.
class RemoteAccessPanel extends StatefulWidget {
  const RemoteAccessPanel({super.key, required this.state});
  final AppState state;

  @override
  State<RemoteAccessPanel> createState() => _RemoteAccessPanelState();
}

class _RemoteAccessPanelState extends State<RemoteAccessPanel> {
  final _cfg = AppConfig.instance;
  late final _token = TextEditingController(text: _cfg.tunnelToken);
  late final _host = TextEditingController(text: _cfg.tunnelHostname);
  late final _bin = TextEditingController(text: _cfg.cloudflaredPath);
  late bool _named = _cfg.tunnelToken.isNotEmpty;
  late TunnelTarget _target = TunnelTarget.fromId(_cfg.tunnelTarget);
  bool _showLog = false;
  String? _detected;

  TunnelManager get _t => widget.state.tunnel;

  @override
  void initState() {
    super.initState();
    TunnelManager.locateBinary().then((p) {
      if (mounted) setState(() => _detected = p ?? '');
    });
  }

  @override
  void dispose() {
    _token.dispose();
    _host.dispose();
    _bin.dispose();
    super.dispose();
  }

  Future<void> _start() async {
    if (_target == TunnelTarget.console && !_named) {
      final ok = await confirmAction(
        context,
        title: 'Publish the full console',
        message: 'Anyone who learns the tunnel address can open the dashboard, '
            'watch the cameras and — while security.require_token is false — '
            'change cameras, fences, models or shut the backend down. '
            'Use "Phone cameras only" unless you need remote viewing.',
        confirmLabel: 'Publish console',
      );
      if (!ok) return;
    }
    await _cfg.setTunnel(
      token: _named ? _token.text : '',
      hostname: _named ? _host.text : '',
      cloudflaredPath: _bin.text,
      target: _target.id,
    );
    await _t.start();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: _t,
      builder: (context, _) {
        final running = _t.isRunning;
        final busy = _t.isBusy;
        final locked = running || busy;
        return Panel(
          title: 'Remote access · Cloudflare Tunnel',
          accent: running ? IbvapColors.green : null,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Publishes this local system on a public HTTPS address so phones on any '
                'network can stream in with a trusted certificate — no firewall rule, '
                'port forwarding or certificate warning. The backend must be running. '
                'The log is also saved to %APPDATA%\\ibvap_app\\tunnel.log.',
                style: TextStyle(color: IbvapColors.muted, fontSize: 11.5, height: 1.35),
              ),
              const SizedBox(height: 14),
              _statusRow(),
              if (_t.notice != null) ...[
                const SizedBox(height: 8),
                Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  const Icon(Icons.info_outline, size: 14, color: IbvapColors.orange),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(_t.notice!,
                        style: const TextStyle(color: IbvapColors.orange, fontSize: 11.5)),
                  ),
                ]),
              ],
              if (running && _t.publicUrl != null) ...[
                const SizedBox(height: 12),
                _published(),
              ],
              const SizedBox(height: 14),
              const Divider(height: 1),
              const SizedBox(height: 12),
              Text('TUNNEL TYPE', style: IbvapText.label(size: 9.5)),
              const SizedBox(height: 6),
              Wrap(spacing: 6, children: [
                _choice('Quick (free, no account)', !_named,
                    locked ? null : () => setState(() => _named = false)),
                _choice('Named (your domain)', _named,
                    locked ? null : () => setState(() => _named = true)),
              ]),
              const SizedBox(height: 6),
              Text(
                _named
                    ? 'Create a tunnel in Cloudflare Zero Trust › Networks › Tunnels, add a '
                        'public hostname (e.g. stream.yourdomain.com) pointing to '
                        '${_target == TunnelTarget.phones ? 'https://localhost:8443 with "No TLS Verify" on' : 'http://localhost:8090'}, '
                        'then paste the tunnel token below.'
                    : 'Gets a random https://….trycloudflare.com address that changes every '
                        'time the tunnel starts. Good for quick phone tests.',
                style: const TextStyle(color: IbvapColors.muted, fontSize: 11),
              ),
              if (_named) ...[
                const SizedBox(height: 10),
                Row(children: [
                  Expanded(
                    child: TextField(
                      controller: _host,
                      enabled: !locked,
                      style: IbvapText.data(size: 12.5),
                      decoration: const InputDecoration(
                          labelText: 'Public hostname', hintText: 'stream.mathswithsd.in'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: TextField(
                      controller: _token,
                      enabled: !locked,
                      obscureText: true,
                      style: IbvapText.data(size: 12.5),
                      decoration: const InputDecoration(
                          labelText: 'Tunnel token', hintText: 'eyJhIjoi…'),
                    ),
                  ),
                ]),
              ],
              const SizedBox(height: 12),
              Text('EXPOSE', style: IbvapText.label(size: 9.5)),
              const SizedBox(height: 6),
              Wrap(spacing: 6, children: [
                for (final t in TunnelTarget.values)
                  _choice(t.label, _target == t,
                      locked ? null : () => setState(() => _target = t),
                      warn: t == TunnelTarget.console),
              ]),
              const SizedBox(height: 12),
              Text('CLOUDFLARED', style: IbvapText.label(size: 9.5)),
              const SizedBox(height: 6),
              TextField(
                controller: _bin,
                enabled: !locked,
                style: IbvapText.data(size: 12),
                decoration: InputDecoration(
                  hintText: 'auto-detect (PATH / Program Files)',
                  helperText: _detected == null
                      ? 'looking for cloudflared…'
                      : _detected!.isEmpty
                          ? 'not found — install with: winget install --id Cloudflare.cloudflared'
                          : 'found: $_detected',
                  helperStyle: TextStyle(
                      color: _detected == '' ? IbvapColors.orange : IbvapColors.muted,
                      fontSize: 10.5),
                ),
              ),
              if (_t.log.isNotEmpty) ...[
                const SizedBox(height: 8),
                TextButton.icon(
                  onPressed: () => setState(() => _showLog = !_showLog),
                  icon: Icon(_showLog ? Icons.expand_less : Icons.expand_more, size: 16),
                  label: Text(_showLog ? 'Hide tunnel log' : 'Show tunnel log'),
                ),
                if (_showLog)
                  Container(
                    height: 160,
                    width: double.infinity,
                    padding: const EdgeInsets.all(8),
                    color: Colors.black,
                    child: SingleChildScrollView(
                      reverse: true,
                      child: SelectableText(_t.log.join('\n'),
                          style: IbvapText.data(size: 10.5, color: IbvapColors.muted)),
                    ),
                  ),
              ],
            ],
          ),
        );
      },
    );
  }

  Widget _statusRow() {
    final (color, label) = switch (_t.status) {
      TunnelStatus.running => (IbvapColors.green, 'TUNNEL UP'),
      TunnelStatus.starting => (IbvapColors.orange, 'CONNECTING…'),
      TunnelStatus.verifying => (IbvapColors.orange, 'CHECKING PUBLIC ADDRESS…'),
      TunnelStatus.failed => (IbvapColors.red, 'FAILED'),
      TunnelStatus.stopped => (IbvapColors.muted, 'OFF'),
    };
    return Row(
      children: [
        Container(width: 8, height: 8, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 8),
        Text(label, style: IbvapText.label(size: 11, color: color)),
        const SizedBox(width: 12),
        if (_t.error != null)
          Expanded(
            child: Text(_t.error!,
                style: const TextStyle(color: IbvapColors.red, fontSize: 11.5)),
          )
        else
          const Spacer(),
        if (_t.isRunning || _t.isBusy)
          OutlinedButton.icon(
            onPressed: _t.stop,
            icon: const Icon(Icons.stop_rounded, size: 16),
            label: const Text('Stop tunnel'),
          )
        else
          FilledButton.icon(
            onPressed: _start,
            icon: const Icon(Icons.public, size: 16),
            label: const Text('START TUNNEL'),
          ),
      ],
    );
  }

  Widget _published() {
    final base = _t.publicUrl!;
    final phone = '$base/cam/0';
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: IbvapColors.tint(IbvapColors.green, 0.07),
        border: Border.all(color: IbvapColors.tint(IbvapColors.green, 0.5)),
        borderRadius: BorderRadius.circular(kRadius),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            color: Colors.white,
            padding: const EdgeInsets.all(6),
            child: QrImageView(data: phone, size: 112, backgroundColor: Colors.white),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('PUBLIC ADDRESS', style: IbvapText.label(size: 9)),
                _link(base),
                const SizedBox(height: 8),
                Text('PHONE CAMERA LINKS', style: IbvapText.label(size: 9)),
                for (final id in const [0, 1]) _link('$base/cam/$id'),
                if (_t.target == TunnelTarget.console) ...[
                  const SizedBox(height: 8),
                  Text('REMOTE DASHBOARD', style: IbvapText.label(size: 9)),
                  _link('$base/monitor'),
                ],
                const SizedBox(height: 6),
                const Text('Scan the QR on a phone for CAM-00. Use /cam/1 for a second phone.',
                    style: TextStyle(color: IbvapColors.muted, fontSize: 10.5)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _link(String url) => Row(
        children: [
          Flexible(
            child: SelectableText(url, style: IbvapText.data(size: 12, color: IbvapColors.text)),
          ),
          IconButton(
            tooltip: 'Copy',
            iconSize: 14,
            visualDensity: VisualDensity.compact,
            color: IbvapColors.muted,
            icon: const Icon(Icons.copy),
            onPressed: () => Clipboard.setData(ClipboardData(text: url)),
          ),
        ],
      );

  Widget _choice(String label, bool on, VoidCallback? onTap, {bool warn = false}) {
    final c = warn ? IbvapColors.orange : IbvapColors.green;
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: on ? IbvapColors.tint(c, 0.16) : Colors.transparent,
          border: Border.all(color: on ? c : IbvapColors.border),
          borderRadius: BorderRadius.circular(kRadius),
        ),
        child: Text(label,
            style: TextStyle(
                color: on ? c : (onTap == null ? IbvapColors.faint : IbvapColors.muted),
                fontSize: 11.5,
                fontWeight: FontWeight.w600)),
      ),
    );
  }
}
