import 'package:flutter/material.dart';

import '../theme.dart';

/// Confirmation for actions that interrupt surveillance or destroy data.
///
/// When [typeToConfirm] is set the operator must type it exactly before the
/// action unlocks — reserved for the highest-consequence actions (stopping
/// the backend), where a stray click must not be enough.
Future<bool> confirmAction(
  BuildContext context, {
  required String title,
  required String message,
  String confirmLabel = 'Confirm',
  bool destructive = true,
  String? typeToConfirm,
}) async {
  final ok = await showDialog<bool>(
    context: context,
    barrierDismissible: false,
    builder: (_) => _ConfirmDialog(
      title: title,
      message: message,
      confirmLabel: confirmLabel,
      destructive: destructive,
      typeToConfirm: typeToConfirm,
    ),
  );
  return ok == true;
}

class _ConfirmDialog extends StatefulWidget {
  const _ConfirmDialog({
    required this.title,
    required this.message,
    required this.confirmLabel,
    required this.destructive,
    required this.typeToConfirm,
  });

  final String title;
  final String message;
  final String confirmLabel;
  final bool destructive;
  final String? typeToConfirm;

  @override
  State<_ConfirmDialog> createState() => _ConfirmDialogState();
}

class _ConfirmDialogState extends State<_ConfirmDialog> {
  final _ctrl = TextEditingController();

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  bool get _unlocked =>
      widget.typeToConfirm == null ||
      _ctrl.text.trim().toUpperCase() == widget.typeToConfirm!.toUpperCase();

  @override
  Widget build(BuildContext context) {
    final accent = widget.destructive ? IbvapColors.red : IbvapColors.orange;
    return AlertDialog(
      titlePadding: EdgeInsets.zero,
      title: Container(
        padding: const EdgeInsets.fromLTRB(20, 14, 20, 12),
        decoration: BoxDecoration(
          border: Border(left: BorderSide(color: accent, width: 3)),
        ),
        child: Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: accent, size: 20),
            const SizedBox(width: 10),
            Expanded(child: Text(widget.title.toUpperCase(),
                style: IbvapText.label(size: 13, color: IbvapColors.text))),
          ],
        ),
      ),
      content: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 440),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(widget.message,
                style: const TextStyle(color: IbvapColors.text, fontSize: 13, height: 1.4)),
            if (widget.typeToConfirm != null) ...[
              const SizedBox(height: 16),
              Text('Type ${widget.typeToConfirm} to confirm',
                  style: IbvapText.label(size: 10)),
              const SizedBox(height: 6),
              TextField(
                controller: _ctrl,
                autofocus: true,
                style: IbvapText.data(size: 13),
                onChanged: (_) => setState(() {}),
                onSubmitted: (_) {
                  if (_unlocked) Navigator.pop(context, true);
                },
              ),
            ],
          ],
        ),
      ),
      actions: [
        TextButton(
          autofocus: widget.typeToConfirm == null,
          onPressed: () => Navigator.pop(context, false),
          style: TextButton.styleFrom(foregroundColor: IbvapColors.muted),
          child: const Text('Cancel'),
        ),
        FilledButton(
          style: FilledButton.styleFrom(
            backgroundColor: accent,
            foregroundColor: widget.destructive ? Colors.white : Colors.black,
            disabledBackgroundColor: IbvapColors.border,
          ),
          onPressed: _unlocked ? () => Navigator.pop(context, true) : null,
          child: Text(widget.confirmLabel.toUpperCase()),
        ),
      ],
    );
  }
}
