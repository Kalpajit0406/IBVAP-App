import 'package:flutter/material.dart';

import '../services/backend_manager.dart';
import '../state/app_state.dart';
import '../theme.dart';
import 'backend_console_dialog.dart';

/// Card displayed when the server is offline, giving the operator instant
/// visibility and 1-click startup of the Python backend with mode selection.
class BackendOfflineCard extends StatelessWidget {
  const BackendOfflineCard({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final backend = state.backend;

    return ListenableBuilder(
      listenable: backend,
      builder: (context, _) {
        final isStarting = backend.isStarting;
        final isFailed = backend.isFailed;

        return Center(
          child: Container(
            constraints: const BoxConstraints(maxWidth: 580),
            margin: const EdgeInsets.all(24),
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: const Color(0xFF1E293B),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: isFailed ? IbvapColors.red : IbvapColors.border,
                width: 1.2,
              ),
              boxShadow: [
                BoxShadow(
                  color: const Color(0x66000000),
                  blurRadius: 16,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
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
                        color: isFailed
                            ? const Color(0x33EF4444)
                            : isStarting
                                ? const Color(0x33F59E0B)
                                : const Color(0x3322C55E),
                        borderRadius: BorderRadius.circular(kRadius),
                      ),
                      child: Icon(
                        isFailed
                            ? Icons.error_outline_rounded
                            : isStarting
                                ? Icons.hourglass_top_rounded
                                : Icons.power_settings_new_rounded,
                        color: isFailed
                            ? IbvapColors.red
                            : isStarting
                                ? IbvapColors.orange
                                : IbvapColors.green,
                        size: 28,
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            isStarting
                                ? 'Starting IBVAP Backend…'
                                : isFailed
                                    ? 'Backend Startup Failed'
                                    : 'IBVAP Backend is Offline',
                            style: const TextStyle(
                              color: IbvapColors.text,
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 3),
                          Text(
                            isStarting
                                ? 'Initializing PyTorch, CUDA, models and server listeners…'
                                : isFailed
                                    ? (backend.lastError ?? 'Check console logs below for error details.')
                                    : 'Turn on the Python analytics pipeline to begin surveillance.',
                            style: TextStyle(
                              color: isFailed ? IbvapColors.red : IbvapColors.muted,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                const Divider(color: IbvapColors.border, height: 1),
                const SizedBox(height: 16),

                // Mode Picker
                const Text(
                  'LAUNCH MODE',
                  style: TextStyle(
                    color: IbvapColors.muted,
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1,
                  ),
                ),
                const SizedBox(height: 8),
                Column(
                  children: BackendMode.values.map((mode) {
                    final selected = backend.mode == mode;
                    return InkWell(
                      borderRadius: BorderRadius.circular(kRadius),
                      onTap: isStarting ? null : () => backend.setMode(mode),
                      child: Container(
                        margin: const EdgeInsets.only(bottom: 6),
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        decoration: BoxDecoration(
                          color: selected ? const Color(0x2222C55E) : const Color(0xFF0F172A),
                          borderRadius: BorderRadius.circular(kRadius),
                          border: Border.all(
                            color: selected ? IbvapColors.green : IbvapColors.border,
                            width: selected ? 1.2 : 0.8,
                          ),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              selected ? Icons.radio_button_checked : Icons.radio_button_off,
                              color: selected ? IbvapColors.green : IbvapColors.muted,
                              size: 16,
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    mode.label,
                                    style: TextStyle(
                                      color: selected ? IbvapColors.green : IbvapColors.text,
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  Text(
                                    mode.description,
                                    style: const TextStyle(
                                      color: IbvapColors.muted,
                                      fontSize: 10.5,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  }).toList(),
                ),
                const SizedBox(height: 16),

                // Action Buttons
                Row(
                  children: [
                    Expanded(
                      child: FilledButton.icon(
                        style: FilledButton.styleFrom(
                          backgroundColor: IbvapColors.green,
                          foregroundColor: Colors.black,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(kRadius)),
                          textStyle: const TextStyle(fontSize: 13, fontWeight: FontWeight.w800),
                        ),
                        icon: isStarting
                            ? const SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(strokeWidth: 2, color: Colors.black),
                              )
                            : const Icon(Icons.play_arrow_rounded, size: 20),
                        label: Text(isStarting ? 'Starting…' : 'Turn On Backend Now'),
                        onPressed: isStarting ? null : () => backend.start(),
                      ),
                    ),
                    const SizedBox(width: 10),
                    OutlinedButton.icon(
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(kRadius)),
                        side: const BorderSide(color: IbvapColors.border),
                      ),
                      icon: const Icon(Icons.terminal_rounded, size: 18, color: IbvapColors.muted),
                      label: const Text('Logs', style: TextStyle(color: IbvapColors.text, fontSize: 12)),
                      onPressed: () => BackendConsoleDialog.show(context, state),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
