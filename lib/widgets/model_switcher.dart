import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../theme.dart';
import '../util/describe_error.dart';

/// The MODEL row — mirrors the web dashboard's hot-swap control. Reads
/// `/api/models` (via [AppState.models]) and POSTs `/api/switch-model`.
class ModelSwitcher extends StatelessWidget {
  const ModelSwitcher({super.key, required this.state});

  final AppState state;

  Future<void> _switch(BuildContext context, String name, bool isRef) async {
    try {
      await state.client.switchModel(
          isRef ? {'weights_ref': name} : {'profile': name});
      await state.refreshNow();
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Switch failed: ${describeError(e)}')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final profiles = (state.models['profiles'] as Map?)?.cast<String, dynamic>() ??
        const {};
    final active = (state.models['active'] ?? state.status?.activeModel ?? '')
        .toString();
    final switching =
        state.models['switching'] == true || (state.status?.switching ?? false);

    if (profiles.isEmpty) {
      return const Text('models: —',
          style: TextStyle(color: IbvapColors.muted, fontSize: 11));
    }

    return Wrap(
      spacing: 6,
      runSpacing: 6,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: [
        const Text('MODEL',
            style: TextStyle(
                color: IbvapColors.muted,
                fontSize: 9,
                letterSpacing: 1,
                fontWeight: FontWeight.w700)),
        for (final e in profiles.entries)
          _chip(
            context,
            name: e.key,
            info: (e.value as Map).cast<String, dynamic>(),
            active: e.key == active,
            switching: switching && e.key == active,
          ),
      ],
    );
  }

  Widget _chip(
    BuildContext context, {
    required String name,
    required Map<String, dynamic> info,
    required bool active,
    required bool switching,
  }) {
    final isRef = info['ref'] == true;
    final label = (info['label'] ?? name).toString();
    final delta = info['delta'];
    final bg = switching
        ? IbvapColors.orange
        : active
            ? IbvapColors.green
            : Colors.transparent;
    final fg = (switching || active) ? Colors.black : IbvapColors.muted;

    return InkWell(
      borderRadius: BorderRadius.circular(kRadius),
      onTap: active ? null : () => _switch(context, name, isRef),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(kRadius),
          border: Border.all(
              color: active || switching
                  ? Colors.transparent
                  : IbvapColors.border),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(label,
                style: TextStyle(
                    color: fg, fontSize: 11, fontWeight: FontWeight.w700)),
            if (isRef && delta is num) ...[
              const SizedBox(width: 5),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                decoration: BoxDecoration(
                  color: const Color(0x3322C55E),
                  borderRadius: BorderRadius.circular(kRadius),
                ),
                child: Text(
                  '${delta >= 0 ? '+' : ''}${(delta * 100).toStringAsFixed(1)} mAP',
                  style: const TextStyle(
                      color: IbvapColors.green,
                      fontSize: 8,
                      fontWeight: FontWeight.w700),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
