import 'package:flutter/material.dart';

import '../models/fence.dart';
import '../models/threat.dart';
import '../models/server_status.dart';
import '../state/app_state.dart';
import '../theme.dart';
import '../util/describe_error.dart';
import '../widgets/backend_offline_card.dart';
import '../widgets/confirm_dialog.dart';
import '../widgets/fence_overlay.dart';
import '../widgets/mjpeg_view.dart';
import '../widgets/panel.dart';
import 'panels/alerts_section.dart';
import 'panels/fences_section.dart';
import 'panels/learning_section.dart';

/// The main workspace: the live mosaic with the fence overlay + draw tools on
/// the left, and the Alerts / Fences / Learning sidebar on the right — the same
/// shape as the web dashboard.
class MonitorScreen extends StatefulWidget {
  const MonitorScreen({super.key, required this.state});
  final AppState state;

  @override
  State<MonitorScreen> createState() => _MonitorScreenState();
}

class _MonitorScreenState extends State<MonitorScreen> {
  String _mode = 'grid';
  final Set<int> _mains = {};
  bool _localLayoutDirty = false;

  void _syncFromStatus(MosaicLayout layout) {
    if (_localLayoutDirty) return;
    _mode = layout.mode;
    _mains
      ..clear()
      ..addAll(layout.mains);
  }

  Future<void> _pushLayout() async {
    _localLayoutDirty = true;
    setState(() {});
    try {
      await widget.state.client.setLayout(
        mode: _mode,
        mains: _mains.toList(),
      );
      await widget.state.refreshNow();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Layout change failed: ${describeError(e)}')));
      }
    }
    _localLayoutDirty = false;
    if (mounted) setState(() {});
  }

  void _setMode(String m, List<int> ids) {
    _mode = m;
    if (m == 'grid') {
      _mains.clear();
    } else if (_mains.isEmpty && ids.isNotEmpty) {
      _mains.addAll(ids.take(4));
    }
    _pushLayout();
  }

  void _toggleMain(int id) {
    if (_mains.contains(id)) {
      _mains.remove(id);
    } else if (_mains.length < 4) {
      _mains.add(id);
    }
    _pushLayout();
  }

  String _camName(int id) {
    for (final d in widget.state.status?.devices ?? const <Map<String, dynamic>>[]) {
      final did = d['cam_id'] ?? d['id'];
      if (did is num && did.toInt() == id) {
        final n = (d['label'] ?? d['name'] ?? '').toString();
        if (n.isNotEmpty && n != camLabel(id)) return '${camLabel(id)} · $n';
      }
    }
    return camLabel(id);
  }

  Future<void> _removeStream(int id) async {
    final name = _camName(id);
    final ok = await confirmAction(
      context,
      title: 'Remove stream',
      message: 'Remove $name from the live picture?\n\n'
          'Its capture is closed, a connected phone is disconnected and the slot '
          'is deleted from the camera configuration. Recorded evidence and '
          'snapshots are kept. Add it again from Cameras to restore it.',
      confirmLabel: 'Remove stream',
    );
    if (!ok || !mounted) return;
    final messenger = ScaffoldMessenger.of(context);
    try {
      await widget.state.client.deleteStream(id);
      _mains.remove(id);
      await widget.state.refreshNow();
      messenger.showSnackBar(SnackBar(content: Text('$name removed.')));
    } catch (e) {
      messenger.showSnackBar(
          SnackBar(content: Text('Could not remove $name: ${describeError(e)}')));
    }
  }

  Future<void> _reconnect(int id) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      await widget.state.client.reconnectCamera(id);
      await widget.state.refreshNow();
      messenger.showSnackBar(
          SnackBar(content: Text('${camLabel(id)} reconnect requested.')));
    } catch (e) {
      messenger.showSnackBar(
          SnackBar(content: Text('Reconnect failed: ${describeError(e)}')));
    }
  }

  /// Per-camera actions, opened from a status chip, the Streams menu or a
  /// right-click on the camera's tile.
  Future<void> _showCamMenu(int id, Offset globalPos) async {
    final overlay = Overlay.of(context).context.findRenderObject() as RenderBox;
    final choice = await showMenu<String>(
      context: context,
      position: RelativeRect.fromRect(
          globalPos & const Size(1, 1), Offset.zero & overlay.size),
      items: [
        PopupMenuItem<String>(
          enabled: false,
          height: 30,
          child: Text(_camName(id).toUpperCase(),
              style: IbvapText.label(size: 10, color: IbvapColors.muted)),
        ),
        if (_mode == 'focus')
          PopupMenuItem(
            value: 'focus',
            child: _menuRow(Icons.center_focus_strong_outlined,
                _mains.contains(id) ? 'Remove from mains' : 'Show as main'),
          ),
        PopupMenuItem(value: 'reconnect', child: _menuRow(Icons.refresh, 'Reconnect')),
        const PopupMenuDivider(),
        PopupMenuItem(
          value: 'remove',
          child: _menuRow(Icons.remove_circle_outline, 'Remove stream…',
              color: IbvapColors.red),
        ),
      ],
    );
    if (!mounted) return;
    switch (choice) {
      case 'focus':
        _toggleMain(id);
      case 'reconnect':
        await _reconnect(id);
      case 'remove':
        await _removeStream(id);
    }
  }

  Widget _menuRow(IconData icon, String label, {Color? color}) => Row(children: [
        Icon(icon, size: 16, color: color ?? IbvapColors.muted),
        const SizedBox(width: 10),
        Text(label, style: TextStyle(color: color ?? IbvapColors.text, fontSize: 13)),
      ]);

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([widget.state, widget.state.fences]),
      builder: (context, _) {
        final s = widget.state.status;
        final layout = s?.layout;
        if (layout != null) _syncFromStatus(layout);

        return Row(
          children: [
            Expanded(child: _stage(s, layout)),
            const VerticalDivider(width: 1),
            SizedBox(width: 344, child: _sidebar()),
          ],
        );
      },
    );
  }

  Widget _stage(ServerStatus? status, MosaicLayout? layout) {
    // Only a confirmed-down link replaces the stage. A degraded link keeps
    // the video up (MjpegView flags its own stall) so one slow poll doesn't
    // tear down and re-open the stream.
    if (widget.state.link == LinkState.offline) {
      return BackendOfflineCard(state: widget.state);
    }

    final ids = layout?.ids ?? const <int>[];
    final client = widget.state.client;

    return Column(
      children: [
        _layoutBar(ids),
        const Divider(height: 1),
        Expanded(
          child: Container(
            color: const Color(0xFF05070A),
            padding: const EdgeInsets.all(8),
            child: ids.isEmpty
                ? const _EmptyStage()
                : Center(
                    child: AspectRatio(
                      aspectRatio: (layout!.canvasW <= 0 || layout.canvasH <= 0)
                          ? 16 / 9
                          : layout.canvasW / layout.canvasH,
                      child: LayoutBuilder(builder: (context, box) {
                        // Right-click a tile for its camera actions. The rect
                        // map is in canvas pixels; the view is scaled to fit.
                        return GestureDetector(
                          behavior: HitTestBehavior.translucent,
                          onSecondaryTapUp: (d) {
                            final tile = layout.tileAtCanvas(
                              d.localPosition.dx / box.maxWidth * layout.canvasW,
                              d.localPosition.dy / box.maxHeight * layout.canvasH,
                            );
                            if (tile != null) _showCamMenu(tile.camId, d.globalPosition);
                          },
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              MjpegView(url: client.streamUrl, fit: BoxFit.fill),
                              FenceOverlay(
                                layout: layout,
                                editor: widget.state.fences,
                                activeByCam: status?.geofenceActive ?? const {},
                              ),
                            ],
                          ),
                        );
                      }),
                    ),
                  ),
          ),
        ),
        _drawBar(),
      ],
    );
  }

  Widget _layoutBar(List<int> ids) {
    return Container(
      height: 42,
      padding: const EdgeInsets.symmetric(horizontal: 10),
      color: IbvapColors.surface,
      child: Row(
        children: [
          Text('LAYOUT', style: IbvapText.label(size: 9, color: IbvapColors.faint)),
          const SizedBox(width: 8),
          _seg('Grid', _mode == 'grid', () => _setMode('grid', ids)),
          const SizedBox(width: 4),
          _seg('Focus', _mode == 'focus', () => _setMode('focus', ids)),
          if (_mode == 'focus') ...[
            const SizedBox(width: 10),
            const Text('mains',
                style: TextStyle(color: IbvapColors.muted, fontSize: 10)),
            const SizedBox(width: 6),
            Expanded(
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: [
                    for (final id in ids)
                      Padding(
                        padding: const EdgeInsets.only(right: 4),
                        child: _seg(
                          'CAM-${id.toString().padLeft(2, '0')}',
                          _mains.contains(id),
                          () => _toggleMain(id),
                          accent: IbvapColors.blue,
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ] else
            const Spacer(),
          const SizedBox(width: 10),
          _camReadout(ids),
          if (ids.isNotEmpty) ...[
            const SizedBox(width: 8),
            Builder(builder: (btnContext) {
              return Tooltip(
                message: 'Stream actions (or right-click a camera tile)',
                child: OutlinedButton.icon(
                  style: OutlinedButton.styleFrom(
                    visualDensity: VisualDensity.compact,
                    padding: const EdgeInsets.symmetric(horizontal: 10),
                    foregroundColor: IbvapColors.muted,
                    side: const BorderSide(color: IbvapColors.border),
                  ),
                  icon: const Icon(Icons.video_settings_outlined, size: 15),
                  label: const Text('Streams', style: TextStyle(fontSize: 11)),
                  onPressed: () => _showStreamsMenu(btnContext, ids),
                ),
              );
            }),
          ],
        ],
      ),
    );
  }

  Future<void> _showStreamsMenu(BuildContext btnContext, List<int> ids) async {
    final box = btnContext.findRenderObject() as RenderBox;
    final overlay = Overlay.of(context).context.findRenderObject() as RenderBox;
    final origin = box.localToGlobal(Offset(0, box.size.height), ancestor: overlay);
    final id = await showMenu<int>(
      context: context,
      position: RelativeRect.fromRect(
          origin & Size(box.size.width, 1), Offset.zero & overlay.size),
      items: [
        PopupMenuItem<int>(
          enabled: false,
          height: 30,
          child: Text('CHOOSE A CAMERA',
              style: IbvapText.label(size: 10, color: IbvapColors.muted)),
        ),
        for (final i in ids)
          PopupMenuItem(value: i, child: _menuRow(Icons.videocam_outlined, _camName(i))),
      ],
    );
    if (id != null && mounted) {
      await _showCamMenu(id, origin + Offset(box.size.width, 0));
    }
  }

  /// Per-camera condition at a glance: colour = risk level, glyphs for
  /// weapon / breach, so the operator sees which tile to look at.
  Widget _camReadout(List<int> ids) {
    final s = widget.state.status;
    if (s == null || ids.isEmpty) return const SizedBox.shrink();
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        for (final id in ids.take(12))
          Builder(builder: (_) {
            final m = s.metaFor(id);
            final level = (m['level'] ?? 'Normal').toString();
            final weapon = m['weapon'] == true || m['armed'] == true;
            final breach = m['breach'] == true;
            final c = weapon || breach || level == 'Critical'
                ? IbvapColors.red
                : level == 'High'
                    ? IbvapColors.orange
                    : IbvapColors.green;
            return Tooltip(
              message: '${camLabel(id)} · ${level.toUpperCase()}'
                  '${weapon ? ' · WEAPON' : ''}${breach ? ' · BREACH' : ''}'
                  ' · P:${m['persons'] ?? 0} V:${m['vehicles'] ?? 0}'
                  '\nClick for camera actions',
              child: GestureDetector(
                onTapUp: (d) => _showCamMenu(id, d.globalPosition),
                onSecondaryTapUp: (d) => _showCamMenu(id, d.globalPosition),
                child: MouseRegion(
                  cursor: SystemMouseCursors.click,
                  child: Container(
                margin: const EdgeInsets.only(left: 4),
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                decoration: BoxDecoration(
                  color: IbvapColors.tint(c, c == IbvapColors.green ? 0.08 : 0.2),
                  border: Border.all(color: IbvapColors.tint(c, 0.7)),
                  borderRadius: BorderRadius.circular(kRadius),
                ),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Text(id.toString().padLeft(2, '0'),
                      style: IbvapText.data(size: 10.5, color: c, weight: FontWeight.w700)),
                  if (weapon) ...[
                    const SizedBox(width: 3),
                    Icon(Icons.gps_fixed, size: 11, color: c),
                  ],
                  if (breach) ...[
                    const SizedBox(width: 3),
                    Icon(Icons.fence_outlined, size: 11, color: c),
                  ],
                ]),
                  ),
                ),
              ),
            );
          }),
      ],
    );
  }

  Widget _seg(String label, bool on, VoidCallback onTap, {Color? accent}) {
    final c = accent ?? IbvapColors.green;
    return InkWell(
      borderRadius: BorderRadius.circular(kRadius),
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: on ? c : Colors.transparent,
          borderRadius: BorderRadius.circular(kRadius),
          border: Border.all(color: on ? Colors.transparent : IbvapColors.border),
        ),
        child: Text(label,
            style: TextStyle(
                color: on ? Colors.black : IbvapColors.muted,
                fontSize: 10.5,
                fontWeight: FontWeight.w700)),
      ),
    );
  }

  Widget _drawBar() {
    final ed = widget.state.fences;
    if (!ed.armed) return const SizedBox.shrink();
    final n = ed.draft.length;
    final need = ed.kind == 'line' ? 2 : 3;
    return Container(
      height: 44,
      color: const Color(0xFF1B140A),
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Row(
        children: [
          const Icon(Icons.edit_location_alt_outlined,
              color: IbvapColors.orange, size: 16),
          const SizedBox(width: 8),
          Text(
            '${ed.editingId != null ? "Editing" : "Drawing"} on '
            'CAM-${ed.draftCam.toString().padLeft(2, '0')} · $n pt${n == 1 ? "" : "s"}'
            '${n < need ? "  (need $need+)" : ""}',
            style: const TextStyle(color: IbvapColors.text, fontSize: 12),
          ),
          const Spacer(),
          TextButton(
            onPressed: n == 0 ? null : ed.undoPoint,
            child: const Text('Undo pt'),
          ),
          TextButton(
            onPressed: ed.canSave && !ed.busy
                ? () async {
                    final err = await ed.save();
                    if (err != null && mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text('Fence rejected: $err')));
                    }
                  }
                : null,
            child: const Text('Save'),
          ),
          TextButton(
            onPressed: ed.cancel,
            style: TextButton.styleFrom(foregroundColor: IbvapColors.muted),
            child: const Text('Cancel'),
          ),
        ],
      ),
    );
  }

  Widget _sidebar() {
    return Container(
      color: IbvapColors.surface,
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          const SectionHeader('Alert log'),
          const Divider(height: 1),
          AlertsSection(state: widget.state),
          const Divider(height: 1),
          const SectionHeader('Virtual fences'),
          const Divider(height: 1),
          FencesSection(state: widget.state),
          const Divider(height: 1),
          const SectionHeader('Continuous learning'),
          const Divider(height: 1),
          LearningSection(state: widget.state),
        ],
      ),
    );
  }
}

class _EmptyStage extends StatelessWidget {
  const _EmptyStage();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.videocam_off_outlined, color: IbvapColors.muted, size: 34),
          SizedBox(height: 12),
          Text('No cameras streaming yet.',
              style: TextStyle(color: IbvapColors.muted, fontSize: 13)),
          SizedBox(height: 6),
          Text('Start the IBVAP server, then a phone or replay feed.',
              style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
        ],
      ),
    );
  }
}
