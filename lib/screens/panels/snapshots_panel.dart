import 'package:flutter/material.dart';

import '../../state/app_state.dart';
import '../../theme.dart';
import '../../widgets/panel.dart';
import '../../util/describe_error.dart';

/// The intrusion snapshots the server wrote (fence breach / weapon /
/// crouch-lying) with thumbnails, camera + reason filters, and a zoomable
/// viewer with an annotated ⇄ raw toggle.
class SnapshotsPanel extends StatefulWidget {
  const SnapshotsPanel({super.key, required this.state});
  final AppState state;

  @override
  State<SnapshotsPanel> createState() => _SnapshotsPanelState();
}

class _SnapshotsPanelState extends State<SnapshotsPanel> {
  List<Map<String, dynamic>> _rows = const [];
  String? _error;
  bool _loading = true;
  int? _camFilter;
  String? _reasonFilter;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final rows = await widget.state.client.snapshots(limit: 120);
      if (!mounted) return;
      setState(() {
        _rows = rows
            .whereType<Map>()
            .map((e) => e.cast<String, dynamic>())
            .toList();
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = describeError(e);
        _loading = false;
      });
    }
  }

  List<Map<String, dynamic>> get _filtered => _rows.where((r) {
        if (_camFilter != null &&
            (r['cam_id'] as num?)?.toInt() != _camFilter) {
          return false;
        }
        if (_reasonFilter != null && r['reason'] != _reasonFilter) {
          return false;
        }
        return true;
      }).toList();

  Color _tagColor(String reason) => switch (reason) {
        'weapon' || 'breach' => IbvapColors.red,
        'posture' => IbvapColors.orange,
        _ => IbvapColors.blue,
      };

  @override
  Widget build(BuildContext context) {
    final cams = {for (final r in _rows) (r['cam_id'] as num).toInt()}.toList()
      ..sort();
    final reasons = {for (final r in _rows) r['reason'].toString()}.toList()
      ..sort();

    return Panel(
      title: 'Snapshots  (${_filtered.length}/${_rows.length})',
      expand: true,
      actions: [
        IconButton(
          onPressed: _loading ? null : _load,
          icon: const Icon(Icons.refresh, size: 16),
          color: IbvapColors.muted,
          tooltip: 'Reload',
        ),
      ],
      padding: EdgeInsets.zero,
      child: Column(
        children: [
          if (_rows.isNotEmpty)
            Container(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
              decoration: const BoxDecoration(
                border:
                    Border(bottom: BorderSide(color: IbvapColors.border)),
              ),
              child: Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [
                  _chip('All cams', _camFilter == null,
                      () => setState(() => _camFilter = null)),
                  for (final c in cams)
                    _chip('CAM-${c.toString().padLeft(2, '0')}',
                        _camFilter == c, () => setState(() => _camFilter = c)),
                  const SizedBox(width: 8),
                  _chip('All', _reasonFilter == null,
                      () => setState(() => _reasonFilter = null)),
                  for (final r in reasons)
                    _chip(r, _reasonFilter == r,
                        () => setState(() => _reasonFilter = r),
                        color: _tagColor(r)),
                ],
              ),
            ),
          Expanded(child: _grid()),
        ],
      ),
    );
  }

  Widget _grid() {
    if (_loading) {
      return const Center(child: CircularProgressIndicator(strokeWidth: 2));
    }
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text('Could not load snapshots\n$_error',
              textAlign: TextAlign.center,
              style: const TextStyle(color: IbvapColors.muted)),
        ),
      );
    }
    final rows = _filtered;
    if (rows.isEmpty) {
      return const ComingSoon(
          'No snapshots.\nThey appear on a fence breach, a weapon, or a crouch / lying.');
    }
    return GridView.builder(
      padding: const EdgeInsets.all(14),
      gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
        maxCrossAxisExtent: 280,
        childAspectRatio: 16 / 11,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
      ),
      itemCount: rows.length,
      itemBuilder: (context, i) => _tile(rows[i]),
    );
  }

  Widget _tile(Map<String, dynamic> r) {
    final file = (r['file'] ?? r['raw_file'])?.toString();
    final reason = (r['reason'] ?? '').toString();
    final detail = (r['detail'] ?? '').toString();
    final ts = (r['ts'] as num?)?.toDouble() ?? 0;
    final when =
        DateTime.fromMillisecondsSinceEpoch((ts * 1000).round()).toLocal();
    return InkWell(
      onTap: file == null ? null : () => _viewer(r),
      child: DecoratedBox(
        decoration: BoxDecoration(
          border: Border.all(color: IbvapColors.border),
          borderRadius: BorderRadius.circular(kRadius),
          color: IbvapColors.surfaceAlt,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: ClipRRect(
                borderRadius:
                    const BorderRadius.vertical(top: Radius.circular(kRadius)),
                child: file == null
                    ? const ColoredBox(color: Colors.black)
                    : Image.network(
                        widget.state.client.snapshotUrl(file).toString(),
                        fit: BoxFit.cover,
                        gaplessPlayback: true,
                        errorBuilder: (_, _, _) => const ColoredBox(
                          color: Colors.black,
                          child: Icon(Icons.broken_image_outlined,
                              color: IbvapColors.muted),
                        ),
                      ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(8, 6, 8, 7),
              child: Row(
                children: [
                  Text('CAM-${r['cam_id'].toString().padLeft(2, '0')}',
                      style: const TextStyle(
                          color: IbvapColors.text,
                          fontSize: 11,
                          fontWeight: FontWeight.w700)),
                  const SizedBox(width: 6),
                  Flexible(
                    child: Text(
                      reason.toUpperCase() +
                          (detail.isNotEmpty ? ' · $detail' : ''),
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                          color: _tagColor(reason),
                          fontSize: 10,
                          fontWeight: FontWeight.w700),
                    ),
                  ),
                  const Spacer(),
                  Text(_hms(when),
                      style: const TextStyle(
                          color: IbvapColors.muted, fontSize: 10)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _viewer(Map<String, dynamic> r) {
    final annotated = (r['file'] ?? r['raw_file'])?.toString();
    final raw = (r['raw_file'] ?? r['file'])?.toString();
    var showRaw = false;
    showDialog<void>(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (context, setLocal) {
          final file = showRaw ? raw : annotated;
          return Dialog(
            backgroundColor: IbvapColors.bg,
            insetPadding: const EdgeInsets.all(28),
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    children: [
                      Text(
                        'CAM-${r['cam_id'].toString().padLeft(2, '0')} · '
                        '${r['reason'].toString().toUpperCase()}'
                        '${(r['detail'] ?? '').toString().isNotEmpty ? " · ${r['detail']}" : ""}',
                        style: const TextStyle(
                            color: IbvapColors.text,
                            fontWeight: FontWeight.w700),
                      ),
                      const Spacer(),
                      if (raw != null && annotated != null && raw != annotated)
                        TextButton(
                          onPressed: () => setLocal(() => showRaw = !showRaw),
                          child: Text(showRaw ? 'Show annotated' : 'Show raw'),
                        ),
                      IconButton(
                        onPressed: () => Navigator.pop(context),
                        icon: const Icon(Icons.close, size: 18),
                        color: IbvapColors.muted,
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Flexible(
                    child: InteractiveViewer(
                      maxScale: 6,
                      child: Image.network(
                        widget.state.client.snapshotUrl(file!).toString(),
                        gaplessPlayback: true,
                      ),
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'sha256 ${(r['sha256'] ?? '—').toString()}',
                    style: const TextStyle(
                        color: IbvapColors.muted, fontSize: 9),
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _chip(String label, bool on, VoidCallback onTap, {Color? color}) {
    final c = color ?? IbvapColors.green;
    return InkWell(
      borderRadius: BorderRadius.circular(kRadius),
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
        decoration: BoxDecoration(
          color: on ? c : Colors.transparent,
          borderRadius: BorderRadius.circular(kRadius),
          border: Border.all(color: on ? Colors.transparent : IbvapColors.border),
        ),
        child: Text(label,
            style: TextStyle(
                color: on ? Colors.black : IbvapColors.muted,
                fontSize: 10,
                fontWeight: FontWeight.w700)),
      ),
    );
  }

  static String _hms(DateTime d) =>
      '${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}:${d.second.toString().padLeft(2, '0')}';
}
