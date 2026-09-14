import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../api/ibvap_client.dart';
import '../../models/threat.dart';
import '../../state/app_state.dart';
import '../../theme.dart';
import '../../util/describe_error.dart';
import '../../widgets/panel.dart';

/// Incident history (SQLite event store) and integrity of the tamper-evident
/// SHA-256 evidence chain — the audit trail an after-action review relies on.
class EvidencePanel extends StatefulWidget {
  const EvidencePanel({super.key, required this.state});
  final AppState state;

  @override
  State<EvidencePanel> createState() => _EvidencePanelState();
}

class _EvidencePanelState extends State<EvidencePanel> {
  Map<String, dynamic>? _verify;
  String? _verifyError;
  bool _verifying = false;

  List<Map<String, dynamic>> _events = const [];
  Map<String, dynamic> _counts = const {};
  bool _eventsAvailable = true;
  String? _eventsError;
  bool _loadingEvents = false;

  List<Map<String, dynamic>> _records = const [];
  String? _recordsError;

  int? _cam;
  String? _level;

  IbvapClient get _client => widget.state.client;

  @override
  void initState() {
    super.initState();
    _loadEvents();
    _loadRecords();
    _runVerify();
  }

  Future<void> _runVerify() async {
    setState(() {
      _verifying = true;
      _verifyError = null;
    });
    try {
      final v = await _client.verifyEvidence();
      if (!mounted) return;
      setState(() => _verify = v);
    } catch (e) {
      if (!mounted) return;
      setState(() => _verifyError = describeError(e));
    } finally {
      if (mounted) setState(() => _verifying = false);
    }
  }

  Future<void> _loadEvents() async {
    setState(() {
      _loadingEvents = true;
      _eventsError = null;
    });
    try {
      final j = await _client.events(limit: 300, camId: _cam, level: _level);
      if (!mounted) return;
      setState(() {
        _events = ((j['events'] as List?) ?? const [])
            .whereType<Map>()
            .map((e) => e.cast<String, dynamic>())
            .toList();
        _counts = (j['counts'] as Map?)?.cast<String, dynamic>() ?? const {};
        _eventsAvailable = j['available'] != false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _eventsError = describeError(e));
    } finally {
      if (mounted) setState(() => _loadingEvents = false);
    }
  }

  Future<void> _loadRecords() async {
    try {
      final r = await _client.evidenceRecent(limit: 60);
      if (!mounted) return;
      setState(() {
        _records = r.whereType<Map>().map((e) => e.cast<String, dynamic>()).toList();
        _recordsError = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _recordsError = describeError(e));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _integrityCard(),
        const SizedBox(height: 12),
        Expanded(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(flex: 3, child: _incidents()),
              const SizedBox(width: 12),
              Expanded(flex: 2, child: _chainRecords()),
            ],
          ),
        ),
      ],
    );
  }

  // ── integrity ──────────────────────────────────────────────────────────
  Widget _integrityCard() {
    final v = _verify;
    final ok = v?['ok'] == true;
    final Color color;
    final String headline;
    final String sub;
    if (_verifying && v == null) {
      color = IbvapColors.muted;
      headline = 'VERIFYING…';
      sub = 'Walking the hash chain from genesis';
    } else if (_verifyError != null) {
      color = IbvapColors.orange;
      headline = 'UNVERIFIED';
      sub = _verifyError!;
    } else if (v == null) {
      color = IbvapColors.muted;
      headline = 'NOT CHECKED';
      sub = '';
    } else if (ok) {
      color = IbvapColors.green;
      headline = 'CHAIN INTACT';
      sub = '${v['records']} records verified · no tampering detected';
    } else {
      color = IbvapColors.red;
      headline = 'INTEGRITY FAILURE';
      sub = 'Chain breaks at record ${v['bad_line']} — evidence after this point '
          'cannot be trusted. Preserve ${v['path']} and escalate.';
    }
    final checked = (v?['checked_at'] as num?)?.toDouble();

    return Container(
      padding: const EdgeInsets.fromLTRB(0, 0, 14, 0),
      decoration: BoxDecoration(
        color: v != null && !ok ? IbvapColors.tint(IbvapColors.red, 0.10) : IbvapColors.surface,
        border: Border.all(color: v != null && !ok ? IbvapColors.red : IbvapColors.border),
        borderRadius: BorderRadius.circular(kRadius),
      ),
      child: IntrinsicHeight(
        child: Row(
          children: [
            Container(width: 4, color: color),
            const SizedBox(width: 14),
            Icon(ok ? Icons.verified_user_outlined : Icons.gpp_maybe_outlined,
                color: color, size: 30),
            const SizedBox(width: 14),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('EVIDENCE CHAIN · SHA-256', style: IbvapText.label(size: 9)),
                    const SizedBox(height: 2),
                    Text(headline,
                        style: IbvapText.data(
                            size: 17, color: color, weight: FontWeight.w800, letterSpacing: 1)),
                    if (sub.isNotEmpty)
                      Text(sub, style: const TextStyle(color: IbvapColors.muted, fontSize: 11.5)),
                    if (v != null && v['tip'] != null) ...[
                      const SizedBox(height: 4),
                      _hashLine('TIP', v['tip'].toString()),
                    ],
                  ],
                ),
              ),
            ),
            Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                FilledButton.icon(
                  onPressed: _verifying ? null : _runVerify,
                  icon: _verifying
                      ? const SizedBox(
                          width: 13, height: 13, child: CircularProgressIndicator(strokeWidth: 1.8))
                      : const Icon(Icons.fact_check_outlined, size: 16),
                  label: const Text('VERIFY NOW'),
                ),
                if (checked != null) ...[
                  const SizedBox(height: 4),
                  Text('checked ${_fmtTs(checked)} · ${v?['elapsed_ms']} ms',
                      style: IbvapText.data(size: 9.5, color: IbvapColors.muted)),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _hashLine(String k, String hash) => Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('$k ', style: IbvapText.label(size: 9, color: IbvapColors.faint)),
          SelectableText(hash, style: IbvapText.data(size: 10.5, color: IbvapColors.muted)),
          IconButton(
            tooltip: 'Copy',
            visualDensity: VisualDensity.compact,
            iconSize: 12,
            color: IbvapColors.muted,
            icon: const Icon(Icons.copy),
            onPressed: () => Clipboard.setData(ClipboardData(text: hash)),
          ),
        ],
      );

  // ── incidents ──────────────────────────────────────────────────────────
  Widget _incidents() {
    final cams = widget.state.status?.cameraIds ?? const <int>[];
    return Panel(
      title: 'Incident log',
      expand: true,
      padding: EdgeInsets.zero,
      actions: [
        for (final l in const ['Critical', 'High'])
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: Text('${l.toUpperCase()} ${_counts[l] ?? 0}',
                style: IbvapText.data(
                    size: 10.5,
                    color: l == 'Critical' ? IbvapColors.red : IbvapColors.orange,
                    weight: FontWeight.w700)),
          ),
        IconButton(
          tooltip: 'Reload',
          iconSize: 16,
          color: IbvapColors.muted,
          icon: const Icon(Icons.refresh),
          onPressed: _loadingEvents ? null : _loadEvents,
        ),
      ],
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
            decoration: const BoxDecoration(
                border: Border(bottom: BorderSide(color: IbvapColors.border))),
            child: Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                _chip('ALL LEVELS', _level == null, () => _setLevel(null)),
                _chip('CRITICAL', _level == 'Critical', () => _setLevel('Critical'),
                    color: IbvapColors.red),
                _chip('HIGH', _level == 'High', () => _setLevel('High'),
                    color: IbvapColors.orange),
                const SizedBox(width: 10),
                _chip('ALL CAMS', _cam == null, () => _setCam(null)),
                for (final c in cams) _chip(camLabel(c), _cam == c, () => _setCam(c)),
              ],
            ),
          ),
          _tableHeader(),
          Expanded(child: _incidentBody()),
        ],
      ),
    );
  }

  void _setLevel(String? l) {
    setState(() => _level = l);
    _loadEvents();
  }

  void _setCam(int? c) {
    setState(() => _cam = c);
    _loadEvents();
  }

  Widget _tableHeader() => Container(
        color: IbvapColors.surfaceAlt,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        child: Row(
          children: [
            SizedBox(width: 138, child: Text('TIME (LOCAL)', style: IbvapText.label(size: 9))),
            SizedBox(width: 64, child: Text('CAM', style: IbvapText.label(size: 9))),
            SizedBox(width: 78, child: Text('LEVEL', style: IbvapText.label(size: 9))),
            SizedBox(width: 50, child: Text('SCORE', style: IbvapText.label(size: 9))),
            SizedBox(width: 60, child: Text('P / V', style: IbvapText.label(size: 9))),
            Expanded(child: Text('DETAIL', style: IbvapText.label(size: 9))),
            SizedBox(width: 90, child: Text('EV HASH', style: IbvapText.label(size: 9))),
          ],
        ),
      );

  Widget _incidentBody() {
    if (_loadingEvents && _events.isEmpty) {
      return const Center(child: CircularProgressIndicator(strokeWidth: 2));
    }
    if (_eventsError != null) return _message(Icons.cloud_off, _eventsError!);
    if (!_eventsAvailable) {
      return _message(Icons.hourglass_empty,
          'Event store not open yet — the inference worker starts it once the pipeline runs.');
    }
    if (_events.isEmpty) {
      return _message(Icons.inbox_outlined, 'No incidents recorded for this filter.');
    }
    return ListView.builder(
      itemCount: _events.length,
      itemExtent: 30,
      itemBuilder: (_, i) {
        final e = _events[i];
        final level = (e['level'] ?? '').toString();
        final color = switch (level) {
          'Critical' => IbvapColors.red,
          'High' => IbvapColors.orange,
          'Plate' => IbvapColors.plate,
          _ => IbvapColors.muted,
        };
        final d = (e['details'] as Map?)?.cast<String, dynamic>() ?? const {};
        final hash = (e['ev_hash'] ?? '').toString();
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          decoration: BoxDecoration(
            color: i.isOdd ? const Color(0x08FFFFFF) : null,
            border: Border(left: BorderSide(color: color, width: 2)),
          ),
          child: Row(
            children: [
              SizedBox(
                  width: 138,
                  child: Text(_fmtTs((e['ts'] as num?)?.toDouble() ?? 0),
                      style: IbvapText.data(size: 11))),
              SizedBox(
                  width: 64,
                  child: Text(camLabel((e['cam_id'] as num?)?.toInt() ?? 0),
                      style: IbvapText.data(size: 11, color: IbvapColors.text))),
              SizedBox(
                  width: 78,
                  child: Text(level.toUpperCase(),
                      style: IbvapText.label(size: 10, color: color))),
              SizedBox(
                  width: 50,
                  child: Text(((e['score'] as num?) ?? 0).toStringAsFixed(0),
                      style: IbvapText.data(size: 11))),
              SizedBox(
                  width: 60,
                  child: Text('${e['persons'] ?? 0} / ${e['vehicles'] ?? 0}',
                      style: IbvapText.data(size: 11, color: IbvapColors.muted))),
              Expanded(
                child: Text(_detail(d),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(color: IbvapColors.muted, fontSize: 11)),
              ),
              SizedBox(
                width: 90,
                child: Tooltip(
                  message: hash.isEmpty ? 'not chained' : hash,
                  child: Text(hash.isEmpty ? '—' : '${hash.substring(0, 10)}…',
                      style: IbvapText.data(size: 10.5, color: IbvapColors.faint)),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  static String _detail(Map<String, dynamic> d) {
    final parts = <String>[];
    if (d['plate'] != null) parts.add('plate ${d['plate']}');
    if (d['type'] != null && d['type'] != 'risk') parts.add(d['type'].toString());
    if (d['weapon'] == true) parts.add('weapon');
    if (d['breach'] == true) parts.add('breach');
    final z = d['breach_zones'];
    if (z is List && z.isNotEmpty) parts.add(z.join(', '));
    final a = d['anomalies'];
    if (a is List && a.isNotEmpty) parts.add(a.join(', '));
    return parts.isEmpty ? '—' : parts.join(' · ');
  }

  // ── chain records ──────────────────────────────────────────────────────
  Widget _chainRecords() {
    return Panel(
      title: 'Chain records (latest)',
      expand: true,
      padding: EdgeInsets.zero,
      actions: [
        IconButton(
          tooltip: 'Reload',
          iconSize: 16,
          color: IbvapColors.muted,
          icon: const Icon(Icons.refresh),
          onPressed: _loadRecords,
        ),
      ],
      child: _recordsError != null
          ? _message(Icons.cloud_off, _recordsError!)
          : _records.isEmpty
              ? _message(Icons.link_off, 'Chain is empty.')
              : ListView.separated(
                  itemCount: _records.length,
                  separatorBuilder: (_, _) => const Divider(height: 1),
                  itemBuilder: (_, i) {
                    final r = _records[i];
                    final ev = (r['event'] as Map?)?.cast<String, dynamic>() ?? const {};
                    final h = (r['hash'] ?? '').toString();
                    final p = (r['prev_hash'] ?? '').toString();
                    return Padding(
                      padding: const EdgeInsets.fromLTRB(12, 7, 8, 7),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Text(_fmtTs((r['timestamp'] as num?)?.toDouble() ?? 0),
                                  style: IbvapText.data(size: 11)),
                              const SizedBox(width: 8),
                              if (ev['cam_id'] != null)
                                Text(camLabel((ev['cam_id'] as num).toInt()),
                                    style: IbvapText.data(size: 11, color: IbvapColors.text)),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                    (ev['type'] ?? ev['level'] ?? 'event').toString().toUpperCase(),
                                    style: IbvapText.label(size: 9.5, color: IbvapColors.blue)),
                              ),
                            ],
                          ),
                          const SizedBox(height: 3),
                          Text('#${h.length >= 16 ? h.substring(0, 16) : h}  ← ${p.length >= 8 ? p.substring(0, 8) : p}',
                              style: IbvapText.data(size: 10, color: IbvapColors.faint)),
                        ],
                      ),
                    );
                  },
                ),
    );
  }

  // ── helpers ────────────────────────────────────────────────────────────
  Widget _message(IconData icon, String text) => Center(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, color: IbvapColors.muted, size: 26),
              const SizedBox(height: 8),
              Text(text,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: IbvapColors.muted, fontSize: 12)),
            ],
          ),
        ),
      );

  Widget _chip(String label, bool on, VoidCallback onTap, {Color? color}) {
    final c = color ?? IbvapColors.green;
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: on ? IbvapColors.tint(c, 0.18) : Colors.transparent,
          borderRadius: BorderRadius.circular(kRadius),
          border: Border.all(color: on ? c : IbvapColors.border),
        ),
        child: Text(label, style: IbvapText.label(size: 9.5, color: on ? c : IbvapColors.muted)),
      ),
    );
  }

  static String _fmtTs(double ts) {
    final d = DateTime.fromMillisecondsSinceEpoch((ts * 1000).round());
    String two(int v) => v.toString().padLeft(2, '0');
    return '${d.year}-${two(d.month)}-${two(d.day)} ${two(d.hour)}:${two(d.minute)}:${two(d.second)}';
  }
}
