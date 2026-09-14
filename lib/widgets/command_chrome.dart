import 'dart:async';

import 'package:flutter/material.dart';

import '../models/alert.dart';
import '../models/threat.dart';
import '../state/app_state.dart';
import '../theme.dart';
import 'backend_control_button.dart';
import 'connection_badge.dart';

// ── date-time group clock ───────────────────────────────────────────────────
class DtgClock extends StatefulWidget {
  const DtgClock({super.key});

  @override
  State<DtgClock> createState() => _DtgClockState();
}

class _DtgClockState extends State<DtgClock> {
  late final Timer _t;

  @override
  void initState() {
    super.initState();
    _t = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _t.cancel();
    super.dispose();
  }

  static const _mon = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                       'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];

  static String _two(int v) => v.toString().padLeft(2, '0');

  @override
  Widget build(BuildContext context) {
    final now = DateTime.now();
    final z = now.toUtc();
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Text('${_two(now.hour)}:${_two(now.minute)}:${_two(now.second)}',
            style: IbvapText.data(size: 15, weight: FontWeight.w700)),
        Text('${_two(z.day)}${_two(z.hour)}${_two(z.minute)}Z ${_mon[z.month - 1]} ${z.year % 100}',
            style: IbvapText.data(size: 9.5, color: IbvapColors.muted, letterSpacing: 0.5)),
      ],
    );
  }
}

// ── command header ──────────────────────────────────────────────────────────
class CommandHeader extends StatelessWidget {
  const CommandHeader({super.key, required this.state, required this.onServerTap});

  final AppState state;
  final VoidCallback onServerTap;

  @override
  Widget build(BuildContext context) {
    final t = state.threat;
    return Container(
      height: 54,
      decoration: const BoxDecoration(
        color: IbvapColors.surface,
        border: Border(bottom: BorderSide(color: IbvapColors.border)),
      ),
      child: Row(
        children: [
          const _Brand(),
          _divider(),
          _ThreatReadout(picture: t),
          _divider(),
          Expanded(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 14),
              child: Row(
                children: [
                  ConnectionBadge(link: state.link, latency: state.latency),
                  const SizedBox(width: 14),
                  _ServerChip(state: state, onTap: onServerTap),
                  if (state.isRetraining) ...[
                    const SizedBox(width: 14),
                    const SizedBox(
                        width: 11,
                        height: 11,
                        child: CircularProgressIndicator(
                            strokeWidth: 1.6, color: IbvapColors.orange)),
                    const SizedBox(width: 6),
                    Text('RETRAINING', style: IbvapText.label(color: IbvapColors.orange)),
                  ],
                ],
              ),
            ),
          ),
          BackendControlButton(state: state),
          const SizedBox(width: 6),
          IconButton(
            tooltip: 'Refresh now (F5)',
            icon: const Icon(Icons.sync, size: 18),
            color: IbvapColors.muted,
            onPressed: state.refreshNow,
          ),
          _divider(),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 16),
            child: DtgClock(),
          ),
        ],
      ),
    );
  }

  static Widget _divider() =>
      Container(width: 1, height: 54, color: IbvapColors.border);
}

class _Brand extends StatelessWidget {
  const _Brand();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Row(
        children: [
          Container(
            width: 30,
            height: 30,
            decoration: BoxDecoration(
              border: Border.all(color: IbvapColors.green, width: 1.4),
              borderRadius: BorderRadius.circular(kRadius),
            ),
            child: const Icon(Icons.shield_outlined, color: IbvapColors.green, size: 18),
          ),
          const SizedBox(width: 10),
          Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              RichText(
                text: TextSpan(
                  style: const TextStyle(
                      fontSize: 15, fontWeight: FontWeight.w800, letterSpacing: 2.5),
                  children: [
                    const TextSpan(text: 'IB', style: TextStyle(color: IbvapColors.text)),
                    const TextSpan(text: 'VAP', style: TextStyle(color: IbvapColors.green)),
                  ],
                ),
              ),
              Text('BORDER SURVEILLANCE C2',
                  style: IbvapText.label(size: 8.5).copyWith(letterSpacing: 1.6)),
            ],
          ),
        ],
      ),
    );
  }
}

class _ThreatReadout extends StatelessWidget {
  const _ThreatReadout({required this.picture});
  final ThreatPicture picture;

  @override
  Widget build(BuildContext context) {
    final lvl = picture.level;
    final reason = picture.reasons.isEmpty
        ? (lvl == ThreatLevel.unknown
            ? 'no live picture'
            : '${picture.camsReporting} cam${picture.camsReporting == 1 ? '' : 's'} reporting')
        : picture.reasons.take(2).join(' · ') +
            (picture.reasons.length > 2 ? '  +${picture.reasons.length - 2}' : '');
    return Tooltip(
      message: picture.reasons.isEmpty ? 'Threat condition' : picture.reasons.join('\n'),
      child: Container(
        width: 250,
        padding: const EdgeInsets.symmetric(horizontal: 14),
        color: lvl == ThreatLevel.critical
            ? IbvapColors.tint(IbvapColors.red, 0.16)
            : Colors.transparent,
        child: Row(
          children: [
            Container(width: 4, height: 34, color: lvl.color),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('THREAT CONDITION', style: IbvapText.label(size: 8.5)),
                  const SizedBox(height: 1),
                  Text(lvl.label,
                      style: IbvapText.data(
                          size: 15, color: lvl.color, weight: FontWeight.w800, letterSpacing: 1)),
                  Text(reason,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(color: IbvapColors.muted, fontSize: 10)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ServerChip extends StatelessWidget {
  const _ServerChip({required this.state, required this.onTap});
  final AppState state;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final host = Uri.tryParse(state.baseUrl)?.authority ?? state.baseUrl;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(kRadius),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          border: Border.all(color: IbvapColors.border),
          borderRadius: BorderRadius.circular(kRadius),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.dns_outlined, size: 13, color: IbvapColors.muted),
            const SizedBox(width: 6),
            Text(host, style: IbvapText.data(size: 11, color: IbvapColors.text)),
            const SizedBox(width: 4),
            const Icon(Icons.edit_outlined, size: 11, color: IbvapColors.faint),
          ],
        ),
      ),
    );
  }
}

// ── alarm banner ────────────────────────────────────────────────────────────
/// Flashing strip for unacknowledged critical alerts. It stays up — and the
/// alarm keeps sounding — until an operator acknowledges.
class AlarmBanner extends StatelessWidget {
  const AlarmBanner({super.key, required this.state, required this.onView});
  final AppState state;
  final void Function(AlertEntry e) onView;

  @override
  Widget build(BuildContext context) {
    final e = state.alerts.latestUnacked;
    // The pulsing child only exists while there is something to signal, so an
    // idle console runs no animation ticker at all.
    if (e == null) return const SizedBox.shrink();
    return _PulsingAlarm(state: state, entry: e, onView: onView);
  }
}

class _PulsingAlarm extends StatefulWidget {
  const _PulsingAlarm({required this.state, required this.entry, required this.onView});
  final AppState state;
  final AlertEntry entry;
  final void Function(AlertEntry e) onView;

  @override
  State<_PulsingAlarm> createState() => _PulsingAlarmState();
}

class _PulsingAlarmState extends State<_PulsingAlarm>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 700))
      ..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final log = widget.state.alerts;
    final e = widget.entry;
    final n = log.unackedCount;
    return AnimatedBuilder(
      animation: _pulse,
      builder: (context, child) => Container(
        height: 38,
        color: Color.lerp(const Color(0xFF4A0E0E), const Color(0xFF7A1414), _pulse.value),
        child: child,
      ),
      child: Row(
        children: [
          Container(width: 5, color: IbvapColors.red),
          const SizedBox(width: 12),
          const Icon(Icons.notification_important, color: Colors.white, size: 18),
          const SizedBox(width: 10),
          Text('ALARM', style: IbvapText.label(size: 11, color: Colors.white)),
          const SizedBox(width: 14),
          Text(e.cam, style: IbvapText.data(size: 12, color: Colors.white, weight: FontWeight.w700)),
          const SizedBox(width: 10),
          Text(e.title, style: IbvapText.label(size: 12, color: const Color(0xFFFFC9C9))),
          const SizedBox(width: 10),
          Expanded(
            child: Text(e.detail,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(color: Color(0xFFFFDADA), fontSize: 12)),
          ),
          Text(_hms(e.at), style: IbvapText.data(size: 11, color: const Color(0xFFFFC9C9))),
          if (n > 1) ...[
            const SizedBox(width: 12),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(kRadius),
              ),
              child: Text('$n PENDING',
                  style: IbvapText.label(size: 9.5, color: const Color(0xFF7A1414))),
            ),
          ],
          const SizedBox(width: 12),
          _bannerButton('VIEW', () => widget.onView(e)),
          const SizedBox(width: 6),
          _bannerButton('ACK', () => log.acknowledge(e), filled: true,
              tooltip: 'Acknowledge this alarm (Ctrl+K)'),
          if (n > 1) ...[
            const SizedBox(width: 6),
            _bannerButton('ACK ALL', log.acknowledgeAll,
                tooltip: 'Acknowledge all pending alarms (Ctrl+Shift+K)'),
          ],
          const SizedBox(width: 10),
        ],
      ),
    );
  }

  Widget _bannerButton(String label, VoidCallback onTap,
      {bool filled = false, String? tooltip}) {
    final b = InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
        decoration: BoxDecoration(
          color: filled ? Colors.white : Colors.transparent,
          border: Border.all(color: Colors.white),
          borderRadius: BorderRadius.circular(kRadius),
        ),
        child: Text(label,
            style: IbvapText.label(
                size: 10.5, color: filled ? const Color(0xFF7A1414) : Colors.white)),
      ),
    );
    return tooltip == null ? b : Tooltip(message: tooltip, child: b);
  }

  static String _hms(DateTime d) =>
      '${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}:${d.second.toString().padLeft(2, '0')}';
}

// ── status footer ───────────────────────────────────────────────────────────
class StatusFooter extends StatefulWidget {
  const StatusFooter({super.key, required this.state});
  final AppState state;

  @override
  State<StatusFooter> createState() => _StatusFooterState();
}

class _StatusFooterState extends State<StatusFooter> {
  late final Timer _t;

  @override
  void initState() {
    super.initState();
    // Re-render the data-age readout between polls, so a stalled link is
    // visible even if nothing else triggers a rebuild.
    _t = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _t.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final st = widget.state;
    final s = st.status;
    final age = st.dataAge;
    final ageColor = age == null
        ? IbvapColors.muted
        : age > const Duration(seconds: 10)
            ? IbvapColors.red
            : st.isStale
                ? IbvapColors.orange
                : IbvapColors.muted;
    final backend = st.backend;
    final up = backend.uptime;

    return Container(
      height: 24,
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: const BoxDecoration(
        color: IbvapColors.surface,
        border: Border(top: BorderSide(color: IbvapColors.border)),
      ),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          children: [
            _item('DATA', age == null ? '—' : _fmtAge(age), color: ageColor),
            _item('CAMS', s == null ? '—' : '${s.camerasLive}/${s.cameraCount} LIVE',
                color: s != null && s.cameraCount > 0 && s.camerasLive < s.cameraCount
                    ? IbvapColors.orange
                    : null),
            _item('COMPUTE', s == null ? '—' : (s.onGpu ? 'GPU ${s.device}' : s.device.toUpperCase()),
                color: s != null && !s.onGpu ? IbvapColors.orange : null),
            _item('MODEL', s?.activeModel ?? '—'),
            _item('FPS', s == null ? '—' : s.fps.toStringAsFixed(0)),
            _item('INFER', s == null ? '—' : '${s.inferenceMs.toStringAsFixed(1)}ms'),
            _item('MODE', s?.inputMode.toUpperCase() ?? '—'),
            _item('BACKEND',
                backend.pid != null
                    ? 'PID ${backend.pid}${up != null ? ' · UP ${_fmtAge(up)}' : ''}'
                    : (st.link == LinkState.online ? 'EXTERNAL' : 'NOT RUNNING')),
            _item('TUNNEL',
                st.tunnel.isRunning
                    ? (Uri.tryParse(st.tunnel.publicUrl ?? '')?.host ?? 'UP')
                    : st.tunnel.isBusy
                        ? 'CONNECTING'
                        : 'OFF',
                color: st.tunnel.isRunning ? IbvapColors.green : null),
            _item('ALARMS', '${st.alerts.unackedCount} PENDING',
                color: st.alerts.unackedCount > 0 ? IbvapColors.red : null),
          ],
        ),
      ),
    );
  }

  Widget _item(String k, String v, {Color? color}) => Padding(
        padding: const EdgeInsets.only(right: 18),
        child: Row(
          children: [
            Text(k, style: IbvapText.label(size: 9, color: IbvapColors.faint)),
            const SizedBox(width: 5),
            Text(v, style: IbvapText.data(size: 10.5, color: color ?? IbvapColors.muted)),
          ],
        ),
      );

  static String _fmtAge(Duration d) {
    if (d.inSeconds < 1) return '<1s';
    if (d.inSeconds < 60) return '${d.inSeconds}s';
    if (d.inMinutes < 60) return '${d.inMinutes}m ${d.inSeconds % 60}s';
    return '${d.inHours}h ${d.inMinutes % 60}m';
  }
}
