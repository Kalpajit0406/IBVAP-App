import 'package:flutter/material.dart';

import '../config/app_config.dart';
import '../state/app_state.dart';
import '../theme.dart';

/// Change the server the console is attached to. Replaces the always-visible
/// address field in the header, where a stray keystroke re-pointed the whole
/// console (and cleared the alert log).
class ServerDialog {
  static Future<void> show(BuildContext context, AppState state) {
    return showDialog<void>(
      context: context,
      builder: (_) => _ServerDialog(state: state),
    );
  }
}

class _ServerDialog extends StatefulWidget {
  const _ServerDialog({required this.state});
  final AppState state;

  @override
  State<_ServerDialog> createState() => _ServerDialogState();
}

class _ServerDialogState extends State<_ServerDialog> {
  late final _ctrl = TextEditingController(text: widget.state.baseUrl);
  String? _error;

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  void _apply() {
    final err = AppConfig.validateBaseUrl(_ctrl.text);
    if (err != null) {
      setState(() => _error = err);
      return;
    }
    Navigator.pop(context);
    if (_ctrl.text.trim() != widget.state.baseUrl) {
      widget.state.setBaseUrl(_ctrl.text);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('SERVER CONNECTION'),
      content: SizedBox(
        width: 420,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('IBVAP HTTP LISTENER', style: IbvapText.label(size: 9.5)),
            const SizedBox(height: 6),
            TextField(
              controller: _ctrl,
              autofocus: true,
              style: IbvapText.data(size: 13),
              decoration: InputDecoration(
                hintText: 'http://host:8090',
                errorText: _error,
                prefixIcon: const Icon(Icons.dns_outlined, size: 16),
              ),
              onChanged: (_) {
                if (_error != null) setState(() => _error = null);
              },
              onSubmitted: (_) => _apply(),
            ),
            const SizedBox(height: 10),
            const Text(
              'Switching servers clears the alert log. Write actions against a '
              'remote server need its API token (Settings › Server).',
              style: TextStyle(color: IbvapColors.muted, fontSize: 11),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => _ctrl.text = AppConfig.defaultBaseUrl,
          style: TextButton.styleFrom(foregroundColor: IbvapColors.muted),
          child: const Text('Use local default'),
        ),
        TextButton(
          onPressed: () => Navigator.pop(context),
          style: TextButton.styleFrom(foregroundColor: IbvapColors.muted),
          child: const Text('Cancel'),
        ),
        FilledButton(onPressed: _apply, child: const Text('CONNECT')),
      ],
    );
  }
}
