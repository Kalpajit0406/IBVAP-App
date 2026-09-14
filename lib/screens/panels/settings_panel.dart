import 'package:flutter/material.dart';

import '../../config/app_config.dart';
import '../../state/app_state.dart';
import '../../theme.dart';
import '../../widgets/backend_actions.dart';
import '../../widgets/backend_console_dialog.dart';
import '../../widgets/panel.dart';
import '../../widgets/remote_access_panel.dart';

class SettingsPanel extends StatefulWidget {
  const SettingsPanel({super.key, required this.state});
  final AppState state;

  @override
  State<SettingsPanel> createState() => _SettingsPanelState();
}

class _SettingsPanelState extends State<SettingsPanel> {
  late final TextEditingController _url =
      TextEditingController(text: widget.state.baseUrl);
  late final TextEditingController _backendDir =
      TextEditingController(text: AppConfig.instance.backendDir);
  late final TextEditingController _pythonPath =
      TextEditingController(text: AppConfig.instance.pythonPath);
  late double _poll = widget.state.pollIntervalMs.toDouble();
  late final TextEditingController _token =
      TextEditingController(text: AppConfig.instance.apiToken);
  bool _showToken = false;
  String? _urlError;

  @override
  void dispose() {
    _token.dispose();
    _url.dispose();
    _backendDir.dispose();
    _pythonPath.dispose();
    super.dispose();
  }

  void _applyUrl() {
    final v = _url.text.trim();
    final err = AppConfig.validateBaseUrl(v);
    setState(() => _urlError = err);
    if (err != null) return;
    widget.state.setBaseUrl(v);
    FocusScope.of(context).unfocus();
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text('Connecting to $v…')));
  }

  void _saveBackendConfig() {
    final bDir = _backendDir.text.trim();
    final py = _pythonPath.text.trim();
    if (bDir.isNotEmpty) AppConfig.instance.setBackendDir(bDir);
    if (py.isNotEmpty) AppConfig.instance.setPythonPath(py);
    FocusScope.of(context).unfocus();
    ScaffoldMessenger.of(context)
        .showSnackBar(const SnackBar(content: Text('Backend settings saved.')));
  }

  @override
  Widget build(BuildContext context) {
    final backend = widget.state.backend;

    return ListenableBuilder(
      listenable: backend,
      builder: (context, _) {
        final isRunning = backend.isRunning;
        final isStarting = backend.isStarting;

        return ListView(
          children: [
            // ── Backend Process Control & Configuration ─────────────────────
            Panel(
              title: 'Backend Process Management',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: isRunning
                              ? const Color(0x3322C55E)
                              : isStarting
                                  ? const Color(0x33F59E0B)
                                  : const Color(0x33EF4444),
                          borderRadius: BorderRadius.circular(kRadius),
                          border: Border.all(
                            color: isRunning
                                ? IbvapColors.green
                                : isStarting
                                    ? IbvapColors.orange
                                    : IbvapColors.red,
                          ),
                        ),
                        child: Text(
                          isRunning
                              ? 'ONLINE • PID ${backend.pid ?? "-"}'
                              : isStarting
                                  ? 'STARTING…'
                                  : 'STOPPED',
                          style: TextStyle(
                            color: isRunning
                                ? IbvapColors.green
                                : isStarting
                                    ? IbvapColors.orange
                                    : IbvapColors.red,
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Text(
                        'Launch Mode: ${backend.mode.label}',
                        style: const TextStyle(color: IbvapColors.muted, fontSize: 12),
                      ),
                      const Spacer(),
                      if (!isRunning && !isStarting)
                        FilledButton.icon(
                          style: FilledButton.styleFrom(
                            backgroundColor: IbvapColors.green,
                            foregroundColor: Colors.black,
                          ),
                          icon: const Icon(Icons.play_arrow_rounded, size: 16),
                          label: const Text('Turn On Backend'),
                          onPressed: () => backend.start(),
                        )
                      else ...[
                        OutlinedButton.icon(
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: IbvapColors.border),
                          ),
                          icon: const Icon(Icons.refresh_rounded, size: 14, color: IbvapColors.muted),
                          label: const Text('Restart', style: TextStyle(color: IbvapColors.text, fontSize: 12)),
                          onPressed: () => BackendActions.restart(context, widget.state),
                        ),
                        const SizedBox(width: 8),
                        FilledButton.icon(
                          style: FilledButton.styleFrom(
                            backgroundColor: IbvapColors.red,
                            foregroundColor: Colors.white,
                          ),
                          icon: const Icon(Icons.stop_rounded, size: 16),
                          label: const Text('Stop'),
                          onPressed: () => BackendActions.stop(context, widget.state),
                        ),
                      ],
                      const SizedBox(width: 8),
                      IconButton(
                        tooltip: 'Open Terminal Logs',
                        icon: const Icon(Icons.terminal_rounded, color: IbvapColors.muted),
                        onPressed: () => BackendConsoleDialog.show(context, widget.state),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  const Divider(color: IbvapColors.border, height: 1),
                  const SizedBox(height: 14),

                  // Directory & Python Paths
                  Row(
                    children: [
                      const Text('Backend Root Directory',
                          style: TextStyle(
                              color: IbvapColors.muted, fontSize: 11)),
                      const SizedBox(width: 8),
                      if (AppConfig.instance.isBundled)
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 1),
                          decoration: BoxDecoration(
                            color: const Color(0x3322C55E),
                            borderRadius: BorderRadius.circular(kRadius),
                          ),
                          child: const Text('BUNDLED',
                              style: TextStyle(
                                  color: IbvapColors.green,
                                  fontSize: 9,
                                  fontWeight: FontWeight.w700)),
                        ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  TextField(
                    controller: _backendDir,
                    style: const TextStyle(color: IbvapColors.text, fontSize: 13),
                    decoration: const InputDecoration(
                        hintText: r'...\backend\app (ships next to the app)'),
                  ),
                  const SizedBox(height: 12),
                  const Text('Python Executable Path',
                      style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
                  const SizedBox(height: 6),
                  TextField(
                    controller: _pythonPath,
                    style: const TextStyle(color: IbvapColors.text, fontSize: 13),
                    decoration: const InputDecoration(
                        hintText: r'...\backend\runtime\Scripts\python.exe'),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'The standalone build ships its own Python + CUDA/torch/'
                    'ultralytics runtime and the trained models under '
                    'backend/ next to the .exe — nothing needs to be '
                    'installed separately. These fields only matter if you '
                    'want to point at a different checkout (e.g. a dev copy).',
                    style: TextStyle(color: IbvapColors.muted, fontSize: 10),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      TextButton(
                        onPressed: () => setState(() {
                          _backendDir.text = AppConfig.defaultBackendDirForDisplay;
                          _pythonPath.text = AppConfig.defaultPythonPathForDisplay;
                        }),
                        child: const Text('Reset to bundled'),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton.icon(
                        icon: const Icon(Icons.save_outlined, size: 16),
                        label: const Text('Save Backend Paths'),
                        onPressed: _saveBackendConfig,
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),

            RemoteAccessPanel(state: widget.state),
            const SizedBox(height: 12),

            // ── Server Connection ───────────────────────────────────────────
            Panel(
              title: 'Server Connection',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Base URL — the IBVAP HTTP listener',
                      style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _url,
                          style: const TextStyle(
                              color: IbvapColors.text, fontSize: 13),
                          decoration: InputDecoration(
                              hintText: 'http://host:8090', errorText: _urlError),
                          onSubmitted: (_) => _applyUrl(),
                        ),
                      ),
                      const SizedBox(width: 8),
                      FilledButton(
                        onPressed: _applyUrl,
                        child: const Text('Connect'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  const Text('default http://127.0.0.1:8090',
                      style: TextStyle(color: IbvapColors.muted, fontSize: 10)),
                  const SizedBox(height: 16),
                  const Divider(height: 1),
                  const SizedBox(height: 14),
                  Text('API WRITE TOKEN', style: IbvapText.label(size: 9.5)),
                  const SizedBox(height: 6),
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _token,
                          obscureText: !_showToken,
                          style: IbvapText.data(size: 12.5),
                          decoration: InputDecoration(
                            hintText: AppConfig.instance.isLoopbackServer
                                ? 'not needed - read automatically from the local backend'
                                : 'paste data/api_token from the remote server',
                            suffixIcon: IconButton(
                              iconSize: 16,
                              tooltip: _showToken ? 'Hide' : 'Show',
                              icon: Icon(_showToken
                                  ? Icons.visibility_off_outlined
                                  : Icons.visibility_outlined),
                              onPressed: () => setState(() => _showToken = !_showToken),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton(
                        onPressed: () async {
                          await AppConfig.instance.setApiToken(_token.text);
                          if (!context.mounted) return;
                          ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text('API token saved.')));
                        },
                        child: const Text('Save token'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  const Text(
                    'The server refuses camera, fence, model and shutdown changes '
                    'from other machines unless this token is sent. Local servers '
                    'need nothing here.',
                    style: TextStyle(color: IbvapColors.muted, fontSize: 10),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),

            // ── Polling ─────────────────────────────────────────────────────
            Panel(
              title: 'Polling',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('/status every ${_poll.round()} ms',
                      style: const TextStyle(
                          color: IbvapColors.text, fontSize: 12)),
                  Slider(
                    value: _poll,
                    min: 300,
                    max: 5000,
                    divisions: 47,
                    label: '${_poll.round()} ms',
                    onChanged: (v) => setState(() => _poll = v),
                    onChangeEnd: (v) =>
                        widget.state.setPollInterval(v.round()),
                  ),
                  const Text(
                      'Lower = snappier UI, more requests. 1000 ms matches the web dashboard.',
                      style: TextStyle(color: IbvapColors.muted, fontSize: 10)),
                ],
              ),
            ),
            const SizedBox(height: 12),

            // ── Console ─────────────────────────────────────────────────────
            Panel(
              title: 'Console',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    dense: true,
                    value: AppConfig.instance.alarmSound,
                    activeThumbColor: IbvapColors.green,
                    title: const Text('Audible alarm',
                        style: TextStyle(color: IbvapColors.text, fontSize: 13)),
                    subtitle: const Text(
                        'Repeats every 6 s while a critical alert is unacknowledged.',
                        style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
                    onChanged: (v) async {
                      await AppConfig.instance.setAlarmSound(v);
                      if (mounted) setState(() {});
                    },
                  ),
                  const SizedBox(height: 8),
                  Text('KEYBOARD', style: IbvapText.label(size: 9.5)),
                  const SizedBox(height: 6),
                  for (final (k, v) in const [
                    ('Ctrl+1 .. Ctrl+8', 'Switch view'),
                    ('Ctrl+K', 'Acknowledge newest alarm'),
                    ('Ctrl+Shift+K', 'Acknowledge all alarms'),
                    ('F5', 'Refresh now'),
                  ])
                    Padding(
                      padding: const EdgeInsets.only(bottom: 3),
                      child: Row(children: [
                        SizedBox(width: 130, child: Text(k, style: IbvapText.data(size: 11.5))),
                        Text(v, style: const TextStyle(color: IbvapColors.muted, fontSize: 11.5)),
                      ]),
                    ),
                ],
              ),
            ),
            const SizedBox(height: 12),

            // ── About ───────────────────────────────────────────────────────
            Panel(
              title: 'About',
              child: DefaultTextStyle(
                style: const TextStyle(color: IbvapColors.muted, fontSize: 11),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: const [
                    Text('IBVAP Command Client',
                        style: TextStyle(fontWeight: FontWeight.bold, color: IbvapColors.text)),
                    SizedBox(height: 4),
                    Text(
                        'AI-Based Intelligent Video Analytics Platform for Border Surveillance.\n'
                        'Supports full backend lifecycle management, real-time mosaic surveillance, '
                        'weapons detection, ANPR license plate parsing, pose estimation, and '
                        'hot-swapping of every trained model.'),
                  ],
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}
