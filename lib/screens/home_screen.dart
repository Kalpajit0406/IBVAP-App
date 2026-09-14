import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../models/alert.dart';
import '../state/app_state.dart';
import '../theme.dart';
import '../widgets/alert_viewer.dart';
import '../widgets/command_chrome.dart';
import '../widgets/server_dialog.dart';
import 'monitor_screen.dart';
import 'panels/anpr_results_panel.dart';
import 'panels/evidence_panel.dart';
import 'panels/models_panel.dart';
import 'panels/overview_panel.dart';
import 'panels/settings_panel.dart';
import 'panels/snapshots_panel.dart';
import 'panels/streams_panel.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.state});

  final AppState state;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _NavIntent extends Intent {
  const _NavIntent(this.index);
  final int index;
}

class _RefreshIntent extends Intent {
  const _RefreshIntent();
}

class _AckIntent extends Intent {
  const _AckIntent({this.all = false});
  final bool all;
}

class _HomeScreenState extends State<HomeScreen> {
  int _index = 0;

  static const _dests = [
    (icon: Icons.grid_view_rounded, label: 'Monitor'),
    (icon: Icons.dashboard_outlined, label: 'Overview'),
    (icon: Icons.videocam_outlined, label: 'Cameras'),
    (icon: Icons.psychology_outlined, label: 'Models'),
    (icon: Icons.photo_library_outlined, label: 'Snapshots'),
    (icon: Icons.directions_car_filled_outlined, label: 'ANPR'),
    (icon: Icons.verified_user_outlined, label: 'Evidence'),
    (icon: Icons.settings_outlined, label: 'Settings'),
  ];

  Widget _body() {
    final s = widget.state;
    Widget padded(Widget w) => Padding(padding: const EdgeInsets.all(14), child: w);
    return switch (_index) {
      0 => MonitorScreen(state: s, key: const ValueKey('monitor')),
      1 => padded(OverviewPanel(state: s)),
      2 => padded(StreamsPanel(state: s, key: const ValueKey('cameras'))),
      3 => padded(ModelsPanel(state: s, key: const ValueKey('models'))),
      4 => padded(SnapshotsPanel(state: s, key: const ValueKey('snapshots'))),
      5 => padded(AnprResultsPanel(state: s, key: const ValueKey('anpr'))),
      6 => padded(EvidencePanel(state: s, key: const ValueKey('evidence'))),
      7 => padded(SettingsPanel(state: s, key: const ValueKey('settings'))),
      _ => const SizedBox.shrink(),
    };
  }

  void _viewAlert(AlertEntry e) {
    if (e.thumbFile != null) {
      AlertViewer.show(context, widget.state, e);
    } else {
      setState(() => _index = 0);
    }
  }

  @override
  Widget build(BuildContext context) {
    final shortcuts = <ShortcutActivator, Intent>{
      for (var i = 0; i < _dests.length; i++)
        SingleActivator(LogicalKeyboardKey(LogicalKeyboardKey.digit1.keyId + i),
            control: true): _NavIntent(i),
      const SingleActivator(LogicalKeyboardKey.f5): const _RefreshIntent(),
      const SingleActivator(LogicalKeyboardKey.keyK, control: true): const _AckIntent(),
      const SingleActivator(LogicalKeyboardKey.keyK, control: true, shift: true):
          const _AckIntent(all: true),
    };

    return Shortcuts(
      shortcuts: shortcuts,
      child: Actions(
        actions: {
          _NavIntent: CallbackAction<_NavIntent>(
              onInvoke: (i) => setState(() => _index = i.index)),
          _RefreshIntent: CallbackAction<_RefreshIntent>(
              onInvoke: (_) => widget.state.refreshNow()),
          _AckIntent: CallbackAction<_AckIntent>(onInvoke: (i) {
            final log = widget.state.alerts;
            if (i.all) {
              log.acknowledgeAll();
            } else if (log.latestUnacked != null) {
              log.acknowledge(log.latestUnacked!);
            }
            return null;
          }),
        },
        child: Focus(
          autofocus: true,
          child: ListenableBuilder(
            listenable: Listenable.merge([widget.state, widget.state.alerts]),
            builder: (context, _) {
              final s = widget.state;
              final pending = s.alerts.unackedCount;
              return Scaffold(
                body: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    CommandHeader(
                      state: s,
                      onServerTap: () => ServerDialog.show(context, s),
                    ),
                    AlarmBanner(state: s, onView: _viewAlert),
                    if (s.link == LinkState.degraded || s.link == LinkState.offline)
                      _LinkWarning(state: s),
                    Expanded(
                      child: Row(
                        children: [
                          NavigationRail(
                            selectedIndex: _index,
                            onDestinationSelected: (i) => setState(() => _index = i),
                            labelType: NavigationRailLabelType.all,
                            minWidth: 72,
                            groupAlignment: -1,
                            destinations: [
                              for (var i = 0; i < _dests.length; i++)
                                NavigationRailDestination(
                                  icon: Tooltip(
                                    message: '${_dests[i].label}  (Ctrl+${i + 1})',
                                    waitDuration: const Duration(milliseconds: 700),
                                    child: Badge(
                                      isLabelVisible: i == 0 && pending > 0,
                                      backgroundColor: IbvapColors.red,
                                      label: Text('$pending'),
                                      child: Icon(_dests[i].icon),
                                    ),
                                  ),
                                  label: Text(_dests[i].label.toUpperCase()),
                                ),
                            ],
                          ),
                          const VerticalDivider(width: 1),
                          Expanded(child: _body()),
                        ],
                      ),
                    ),
                    StatusFooter(state: s),
                  ],
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}

/// Explicit notice that the picture on screen is not current.
class _LinkWarning extends StatelessWidget {
  const _LinkWarning({required this.state});
  final AppState state;

  @override
  Widget build(BuildContext context) {
    final offline = state.link == LinkState.offline;
    final color = offline ? IbvapColors.red : IbvapColors.orange;
    final age = state.dataAge;
    return Container(
      height: 28,
      padding: const EdgeInsets.symmetric(horizontal: 14),
      decoration: BoxDecoration(
        color: IbvapColors.tint(color, 0.12),
        border: Border(bottom: BorderSide(color: IbvapColors.tint(color, 0.5))),
      ),
      child: Row(
        children: [
          Icon(offline ? Icons.link_off : Icons.network_check, size: 15, color: color),
          const SizedBox(width: 8),
          Text(offline ? 'LINK DOWN' : 'LINK DEGRADED',
              style: IbvapText.label(size: 10, color: color)),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              '${state.lastError ?? 'No response from server'}'
              '${age != null ? ' · displayed data is ${age.inSeconds}s old and may not reflect the current situation' : ''}',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(color: IbvapColors.text, fontSize: 11.5),
            ),
          ),
          TextButton(onPressed: state.refreshNow, child: const Text('RETRY')),
        ],
      ),
    );
  }
}
