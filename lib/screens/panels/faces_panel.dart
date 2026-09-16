import 'package:flutter/material.dart';

import '../../state/app_state.dart';
import '../../theme.dart';
import '../../widgets/panel.dart';
import '../../util/describe_error.dart';

/// Event-triggered face-recognition results: a small burst of pictures per
/// person arrival, checked against the watchlist gallery (models/face/
/// gallery.json) — matched identity or "unknown" — from data/face_results/,
/// separate from the intrusion Snapshots panel and the ANPR results panel.
class FacesPanel extends StatefulWidget {
  const FacesPanel({super.key, required this.state});
  final AppState state;

  @override
  State<FacesPanel> createState() => _FacesPanelState();
}

class _FacesPanelState extends State<FacesPanel> {
  List<Map<String, dynamic>> _rows = const [];
  String? _error;
  bool _loading = true;
  int? _camFilter;
  bool? _watchlistFilter; // null = all, true = matched, false = unknown

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
      final rows = await widget.state.client.faceResults(limit: 120);
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
        if (_watchlistFilter != null &&
            (r['on_watchlist'] == true) != _watchlistFilter) {
          return false;
        }
        return true;
      }).toList();

  @override
  Widget build(BuildContext context) {
    final cams = {for (final r in _rows) (r['cam_id'] as num).toInt()}.toList()
      ..sort();
    final matched = _rows.where((r) => r['on_watchlist'] == true).length;

    return Panel(
      title: 'FACES  (${_filtered.length}/${_rows.length})'
          '${matched > 0 ? '  ·  $matched matched' : ''}',
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
                  _chip('All', _watchlistFilter == null,
                      () => setState(() => _watchlistFilter = null)),
                  _chip('Matched', _watchlistFilter == true,
                      () => setState(() => _watchlistFilter = true),
                      color: IbvapColors.red),
                  _chip('Unknown', _watchlistFilter == false,
                      () => setState(() => _watchlistFilter = false)),
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
          child: Text('Could not load face results\n$_error',
              textAlign: TextAlign.center,
              style: const TextStyle(color: IbvapColors.muted)),
        ),
      );
    }
    final rows = _filtered;
    if (rows.isEmpty) {
      return const ComingSoon(
          'No face results yet.\nThey appear the moment a person is detected on a stream.');
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
    final files = ((r['files'] as List?) ?? const []).cast<String>();
    final file = files.isNotEmpty ? files.first : null;
    final onWatchlist = r['on_watchlist'] == true;
    final name = (r['matched_name'] as String?);
    final sim = (r['similarity'] as num?)?.toDouble() ?? 0;
    final ts = (r['ts'] as num?)?.toDouble() ?? 0;
    final when =
        DateTime.fromMillisecondsSinceEpoch((ts * 1000).round()).toLocal();
    return InkWell(
      onTap: file == null ? null : () => _viewer(r),
      child: DecoratedBox(
        decoration: BoxDecoration(
          border: Border.all(
              color: onWatchlist ? IbvapColors.red : IbvapColors.border,
              width: onWatchlist ? 1.5 : 1),
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
                        widget.state.client.faceSnapUrl(file).toString(),
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
                          onWatchlist
                              ? (name?.isNotEmpty == true ? name! : 'MATCH')
                              : 'unknown',
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                              color: onWatchlist
                                  ? IbvapColors.red
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
                  if (onWatchlist)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text('sim ${(sim * 100).round()}%  ·  '
                              '${r['votes']}/${r['of']} votes',
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
    final files = ((r['files'] as List?) ?? const []).cast<String>();
    if (files.isEmpty) return;
    final onWatchlist = r['on_watchlist'] == true;
    final name = (r['matched_name'] as String?);
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
                      '${onWatchlist ? (name?.isNotEmpty == true ? name : 'MATCH') : 'unknown'}',
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
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: [
                      for (final f in files)
                        Padding(
                          padding: const EdgeInsets.only(right: 6),
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(kRadius),
                            child: Image.network(
                              widget.state.client.faceSnapUrl(f).toString(),
                              height: 220,
                              gaplessPlayback: true,
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 6),
              Text(
                'similarity ${(((r['similarity'] as num?)?.toDouble() ?? 0) * 100).round()}%'
                '  ·  votes ${r['votes']}/${r['of']}'
                '  ·  sha256 ${((r['sha256s'] as List?)?.isNotEmpty == true ? (r['sha256s'] as List).first : '—').toString()}',
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

  static String _hms(DateTime d) =>
      '${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}:${d.second.toString().padLeft(2, '0')}';
}
