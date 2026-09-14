import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/backend_manager.dart';
import '../state/app_state.dart';
import '../theme.dart';
import 'backend_actions.dart';

class BackendConsoleDialog extends StatefulWidget {
  const BackendConsoleDialog({super.key, required this.state});

  final AppState state;

  static void show(BuildContext context, AppState state) {
    showDialog(
      context: context,
      builder: (_) => BackendConsoleDialog(state: state),
    );
  }

  @override
  State<BackendConsoleDialog> createState() => _BackendConsoleDialogState();
}

class _BackendConsoleDialogState extends State<BackendConsoleDialog> {
  final ScrollController _scrollController = ScrollController();
  bool _autoScroll = true;

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    if (!_autoScroll || !_scrollController.hasClients) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final backend = widget.state.backend;

    return ListenableBuilder(
      listenable: Listenable.merge([backend, backend.logRevision]),
      builder: (context, _) {
        _scrollToBottom();
        final logs = backend.logs;
        final isRunning = backend.isRunning;
        final isStarting = backend.isStarting;

        return Dialog(
          backgroundColor: const Color(0xFF0F172A),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(kRadius),
            side: const BorderSide(color: IbvapColors.border),
          ),
          insetPadding: const EdgeInsets.symmetric(horizontal: 24, vertical: 24),
          child: SizedBox(
            width: 900,
            height: 620,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // ── Header ─────────────────────────────────────────────────
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: const BoxDecoration(
                    color: Color(0xFF1E293B),
                    borderRadius: BorderRadius.vertical(top: Radius.circular(12)),
                    border: Border(bottom: BorderSide(color: IbvapColors.border)),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.terminal_rounded, size: 20, color: IbvapColors.green),
                      const SizedBox(width: 8),
                      const Text(
                        'IBVAP Backend Console',
                        style: TextStyle(
                          color: IbvapColors.text,
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
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
                            width: 0.8,
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
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                      const Spacer(),
                      // Mode Selector Dropdown
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8),
                        decoration: BoxDecoration(
                          color: const Color(0xFF0F172A),
                          borderRadius: BorderRadius.circular(kRadius),
                          border: Border.all(color: IbvapColors.border),
                        ),
                        child: DropdownButtonHideUnderline(
                          child: DropdownButton<BackendMode>(
                            value: backend.mode,
                            isDense: true,
                            dropdownColor: const Color(0xFF1E293B),
                            style: const TextStyle(color: IbvapColors.text, fontSize: 12),
                            items: BackendMode.values.map((m) {
                              return DropdownMenuItem(
                                value: m,
                                child: Text(m.label),
                              );
                            }).toList(),
                            onChanged: isRunning || isStarting
                                ? null
                                : (m) {
                                    if (m != null) backend.setMode(m);
                                  },
                          ),
                        ),
                      ),
                      const SizedBox(width: 10),
                      // Start/Stop/Restart Actions
                      if (!isRunning && !isStarting)
                        FilledButton.icon(
                          style: FilledButton.styleFrom(
                            backgroundColor: IbvapColors.green,
                            foregroundColor: Colors.black,
                            visualDensity: VisualDensity.compact,
                          ),
                          icon: const Icon(Icons.play_arrow_rounded, size: 16),
                          label: const Text('Start'),
                          onPressed: () => backend.start(),
                        )
                      else ...[
                        OutlinedButton.icon(
                          style: OutlinedButton.styleFrom(
                            visualDensity: VisualDensity.compact,
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
                            visualDensity: VisualDensity.compact,
                          ),
                          icon: const Icon(Icons.stop_rounded, size: 16),
                          label: const Text('Stop'),
                          onPressed: () => BackendActions.stop(context, widget.state),
                        ),
                      ],
                      const SizedBox(width: 8),
                      IconButton(
                        tooltip: 'Close',
                        icon: const Icon(Icons.close_rounded, size: 18, color: IbvapColors.muted),
                        onPressed: () => Navigator.pop(context),
                      ),
                    ],
                  ),
                ),

                // ── Terminal Body ──────────────────────────────────────────
                Expanded(
                  child: Container(
                    color: const Color(0xFF020617),
                    child: logs.isEmpty
                        ? const Center(
                            child: Text(
                              'Terminal ready. Click "Start" to run the IBVAP Python backend.',
                              style: TextStyle(color: IbvapColors.muted, fontSize: 13, fontFamily: 'monospace'),
                            ),
                          )
                        : ListView.builder(
                            controller: _scrollController,
                            padding: const EdgeInsets.all(12),
                            itemCount: logs.length,
                            itemBuilder: (context, index) {
                              final line = logs[index];
                              Color color = const Color(0xFF94A3B8);
                              if (line.contains('ERROR') || line.contains('❌') || line.contains('failed')) {
                                color = const Color(0xFFF87171);
                              } else if (line.contains('WARNING') || line.contains('⚠️') || line.contains('STARTING')) {
                                color = const Color(0xFFFBBF24);
                              } else if (line.contains('ONLINE') || line.contains('✔') || line.contains('ready')) {
                                color = const Color(0xFF4ADE80);
                              } else if (line.contains('▶') || line.contains('━━━━━━━━━━')) {
                                color = const Color(0xFF38BDF8);
                              }

                              return Text(
                                line,
                                style: TextStyle(
                                  color: color,
                                  fontSize: 11.5,
                                  fontFamily: 'monospace',
                                  height: 1.35,
                                ),
                              );
                            },
                          ),
                  ),
                ),

                // ── Footer ─────────────────────────────────────────────────
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: const BoxDecoration(
                    color: Color(0xFF1E293B),
                    borderRadius: BorderRadius.vertical(bottom: Radius.circular(12)),
                    border: Border(top: BorderSide(color: IbvapColors.border)),
                  ),
                  child: Row(
                    children: [
                      Text(
                        '${logs.length} lines logged',
                        style: const TextStyle(color: IbvapColors.muted, fontSize: 11),
                      ),
                      const Spacer(),
                      Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Checkbox(
                            value: _autoScroll,
                            activeColor: IbvapColors.green,
                            checkColor: Colors.black,
                            visualDensity: VisualDensity.compact,
                            onChanged: (v) => setState(() => _autoScroll = v ?? true),
                          ),
                          const Text('Auto-scroll', style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
                        ],
                      ),
                      const SizedBox(width: 8),
                      TextButton.icon(
                        icon: const Icon(Icons.copy_rounded, size: 14, color: IbvapColors.muted),
                        label: const Text('Copy Logs', style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
                        onPressed: logs.isEmpty
                            ? null
                            : () {
                                Clipboard.setData(ClipboardData(text: backend.logsText));
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(content: Text('Console logs copied to clipboard')),
                                );
                              },
                      ),
                      const SizedBox(width: 4),
                      TextButton.icon(
                        icon: const Icon(Icons.delete_sweep_outlined, size: 14, color: IbvapColors.muted),
                        label: const Text('Clear', style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
                        onPressed: logs.isEmpty ? null : backend.clearLogs,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
