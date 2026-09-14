import 'package:flutter/material.dart';

import '../services/backend_manager.dart';
import '../state/app_state.dart';
import '../theme.dart';
import 'backend_actions.dart';
import 'backend_console_dialog.dart';

/// Top bar control widget that enables the operator to turn on, monitor,
/// restart, or stop the IBVAP Python backend with one click.
class BackendControlButton extends StatelessWidget {
  const BackendControlButton({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final backend = state.backend;

    return ListenableBuilder(
      listenable: backend,
      builder: (context, _) {
        final isOnline = state.link == LinkState.online;
        final isRunning = backend.isRunning;
        final isStarting = backend.isStarting;

        // 1. When Starting
        if (isStarting) {
          return InkWell(
            borderRadius: BorderRadius.circular(kRadius),
            onTap: () => BackendConsoleDialog.show(context, state),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0x22F59E0B),
                borderRadius: BorderRadius.circular(kRadius),
                border: Border.all(color: IbvapColors.orange, width: 0.8),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SizedBox(
                    width: 12,
                    height: 12,
                    child: CircularProgressIndicator(strokeWidth: 2, color: IbvapColors.orange),
                  ),
                  SizedBox(width: 8),
                  Text(
                    'Starting Backend…',
                    style: TextStyle(color: IbvapColors.orange, fontSize: 11, fontWeight: FontWeight.w700),
                  ),
                ],
              ),
            ),
          );
        }

        // 2. When Running or Online
        if (isRunning || isOnline) {
          return PopupMenuButton<String>(
            tooltip: 'Backend Controls',
            color: const Color(0xFF1E293B),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(kRadius),
              side: const BorderSide(color: IbvapColors.border),
            ),
            onSelected: (val) {
              switch (val) {
                case 'console':
                  BackendConsoleDialog.show(context, state);
                  break;
                case 'restart':
                  BackendActions.restart(context, state);
                  break;
                case 'stop':
                  BackendActions.stop(context, state);
                  break;
              }
            },
            itemBuilder: (_) => [
              PopupMenuItem(
                value: 'console',
                child: Row(
                  children: const [
                    Icon(Icons.terminal_rounded, size: 16, color: IbvapColors.green),
                    SizedBox(width: 8),
                    Text('View Console Logs', style: TextStyle(color: IbvapColors.text, fontSize: 12)),
                  ],
                ),
              ),
              PopupMenuItem(
                value: 'restart',
                child: Row(
                  children: const [
                    Icon(Icons.refresh_rounded, size: 16, color: IbvapColors.orange),
                    SizedBox(width: 8),
                    Text('Restart Backend', style: TextStyle(color: IbvapColors.text, fontSize: 12)),
                  ],
                ),
              ),
              PopupMenuItem(
                value: 'stop',
                child: Row(
                  children: const [
                    Icon(Icons.stop_circle_outlined, size: 16, color: IbvapColors.red),
                    SizedBox(width: 8),
                    Text('Stop Backend', style: TextStyle(color: IbvapColors.red, fontSize: 12)),
                  ],
                ),
              ),
            ],
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0x1F22C55E),
                borderRadius: BorderRadius.circular(kRadius),
                border: Border.all(color: const Color(0x5522C55E), width: 0.8),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.circle, size: 8, color: IbvapColors.green),
                  const SizedBox(width: 6),
                  Text(
                    isRunning ? 'Backend (PID ${backend.pid})' : 'Backend Online',
                    style: const TextStyle(
                      color: IbvapColors.green,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(width: 4),
                  const Icon(Icons.arrow_drop_down, size: 16, color: IbvapColors.green),
                ],
              ),
            ),
          );
        }

        // 3. When Stopped / Offline -> Turn On Backend
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            FilledButton.icon(
              style: FilledButton.styleFrom(
                backgroundColor: IbvapColors.green,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(kRadius)),
                textStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w800),
              ),
              icon: const Icon(Icons.play_arrow_rounded, size: 18),
              label: const Text('Turn On Backend'),
              onPressed: () {
                backend.start();
                BackendConsoleDialog.show(context, state);
              },
            ),
            const SizedBox(width: 4),
            PopupMenuButton<BackendMode>(
              tooltip: 'Choose Startup Mode',
              color: const Color(0xFF1E293B),
              icon: const Icon(Icons.arrow_drop_down_circle_outlined, size: 18, color: IbvapColors.green),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(kRadius),
                side: const BorderSide(color: IbvapColors.border),
              ),
              onSelected: (mode) {
                backend.start(mode: mode);
                BackendConsoleDialog.show(context, state);
              },
              itemBuilder: (_) => BackendMode.values.map((m) {
                final selected = m == backend.mode;
                return PopupMenuItem(
                  value: m,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Row(
                        children: [
                          if (selected)
                            const Icon(Icons.check, size: 14, color: IbvapColors.green)
                          else
                            const SizedBox(width: 14),
                          const SizedBox(width: 6),
                          Text(
                            m.label,
                            style: TextStyle(
                              color: selected ? IbvapColors.green : IbvapColors.text,
                              fontSize: 12,
                              fontWeight: selected ? FontWeight.bold : FontWeight.normal,
                            ),
                          ),
                        ],
                      ),
                      Padding(
                        padding: const EdgeInsets.only(left: 20, top: 2),
                        child: Text(
                          m.description,
                          style: const TextStyle(color: IbvapColors.muted, fontSize: 10),
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
            const SizedBox(width: 4),
            IconButton(
              tooltip: 'Backend Console',
              icon: const Icon(Icons.terminal_rounded, size: 18, color: IbvapColors.muted),
              onPressed: () => BackendConsoleDialog.show(context, state),
            ),
          ],
        );
      },
    );
  }
}
