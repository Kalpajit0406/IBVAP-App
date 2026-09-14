import 'package:flutter/material.dart';

import '../../state/app_state.dart';
import '../../theme.dart';
import '../../widgets/panel.dart';
import '../../util/describe_error.dart';

/// Event-triggered ANPR results: one picture per vehicle arrival, with its
/// plate number and vehicle type (Car / Pickup truck / Truck / Jeep /
/// 2-wheeler / Tanker / Van / Auto), from data/anpr_results/ — separate from
/// the intrusion Snapshots panel (fence breach / weapon / posture).
class AnprResultsPanel extends StatefulWidget {
  const AnprResultsPanel({super.key, required this.state});
  final AppState state;

  @override
  State<AnprResultsPanel> createState() => _AnprResultsPanelState();
}

class _AnprResultsPanelState extends State<AnprResultsPanel> {
  List<Map<String, dynamic>> _rows = const [];
  String? _error;
  bool _loading = true;
  int? _camFilter;
  String? _typeFilter;

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
      final rows = await widget.state.client.anprResults(limit: 120);
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
        if (_typeFilter != null && r['vehicle_type'] != _typeFilter) {
          return false;
        }
        return true;
      }).toList();

  @override
  Widget build(BuildContext context) {
    final cams = {for (final r in _rows) (r['cam_id'] as num).toInt()}.toList()
      ..sort();
    final types = {
      for (final r in _rows)
        if ((r['vehicle_type'] as String?)?.isNotEmpty ?? false)
          r['vehicle_type'].toString(),
    }.toList()
      ..sort();

    return Panel(
      title: 'ANPR  (${_filtered.length}/${_rows.length})',
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
                  _chip('All types', _typeFilter == null,
                      () => setState(() => _typeFilter = null)),
                  for (final t in types)
                    _chip(_typeLabel(t), _typeFilter == t,
                        () => setState(() => _typeFilter = t)),
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
          child: Text('Could not load ANPR results\n$_error',
              textAlign: TextAlign.center,
              style: const TextStyle(color: IbvapColors.muted)),
        ),
      );
    }
    final rows = _filtered;
    if (rows.isEmpty) {
      return const ComingSoon(
          'No ANPR results yet.\nThey appear the moment a vehicle is detected on a stream.');
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
    final file = (r['file'] as String?);
    final plate = (r['plate'] as String?);
    final vType = (r['vehicle_type'] as String?);
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
                        widget.state.client.anprSnapUrl(file).toString(),
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
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text('CAM-${r['cam_id'].toString().padLeft(2, '0')}',
                          style: const TextStyle(
                              color: IbvapColors.text,
                              fontSize: 11,
                              fontWeight: FontWeight.w700)),
                      const SizedBox(width: 6),
                      Flexible(
                        child: Text(
                          plate?.isNotEmpty == true ? plate! : 'unread',
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                              color: plate?.isNotEmpty == true
                                  ? IbvapColors.green
                                  : IbvapColors.muted,
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
                  if (vType?.isNotEmpty == true)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(_typeLabel(vType!),
                          style: const TextStyle(
                              color: IbvapColors.blue,
                              fontSize: 10,
                              fontWeight: FontWeight.w600)),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _viewer(Map<String, dynamic> r) {
    final file = r['file'] as String?;
    if (file == null) return;
    showDialog<void>(
      context: context,
      builder: (_) => Dialog(
        backgroundColor: IbvapColors.bg,
        insetPadding: const EdgeInsets.all(28),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      'CAM-${r['cam_id'].toString().padLeft(2, '0')} · '
                      '${(r['plate'] as String?)?.isNotEmpty == true ? r['plate'] : 'plate unread'}'
                      '${(r['vehicle_type'] as String?)?.isNotEmpty == true ? " · ${_typeLabel(r['vehicle_type'])}" : ""}',
                      style: const TextStyle(
                          color: IbvapColors.text,
                          fontWeight: FontWeight.w700),
                    ),
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
                    widget.state.client.anprSnapUrl(file).toString(),
                    gaplessPlayback: true,
                  ),
                ),
              ),
              const SizedBox(height: 6),
              Text(
                'plate conf ${(((r['plate_conf'] as num?)?.toDouble() ?? 0) * 100).round()}%'
                '  ·  type conf ${(((r['vehicle_type_conf'] as num?)?.toDouble() ?? 0) * 100).round()}%'
                '${r['vehicle_type_fallback'] == true ? "  (coarse fallback — no trained classifier yet)" : ""}'
                '  ·  sha256 ${(r['sha256'] ?? '—').toString()}',
                style:
                    const TextStyle(color: IbvapColors.muted, fontSize: 9),
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
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

  static String _typeLabel(String t) => switch (t) {
        'two_wheeler' => '2-wheeler',
        'pickup_truck' => 'Pickup truck',
        'auto_rickshaw' => 'Auto',
        'unknown' => 'Unknown',
        _ => t.isEmpty ? t : '${t[0].toUpperCase()}${t.substring(1)}',
      };

  static String _hms(DateTime d) =>
      '${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}:${d.second.toString().padLeft(2, '0')}';
}
