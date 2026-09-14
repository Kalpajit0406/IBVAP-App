import 'package:flutter/material.dart';

import '../state/app_state.dart';
import 'confirm_dialog.dart';

/// Stop / restart the backend with a confirmation. Every entry point (header
/// menu, Settings, console dialog) goes through here so none of them can
/// take cameras off-line on a single click.
class BackendActions {
  static Future<void> stop(BuildContext context, AppState state) async {
    final cams = state.status?.camerasLive ?? 0;
    final ok = await confirmAction(
      context,
      title: 'Stop surveillance backend',
      message: 'This halts detection, recording and alerting on every camera'
          '${cams > 0 ? ' ($cams currently live)' : ''}. '
          'No events will be captured until it is started again.',
      confirmLabel: 'Stop backend',
      typeToConfirm: 'STOP',
    );
    if (ok) await state.backend.stop();
  }

  static Future<void> restart(BuildContext context, AppState state) async {
    final ok = await confirmAction(
      context,
      title: 'Restart backend',
      message: 'Detection and alerting pause for the duration of the restart '
          '(typically 10–30 s while models reload).',
      confirmLabel: 'Restart',
      destructive: false,
    );
    if (ok) await state.backend.restart();
  }
}
