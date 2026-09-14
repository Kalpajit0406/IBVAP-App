import 'package:flutter/material.dart';

import '../../models/alert.dart';
import '../../state/app_state.dart';
import '../../theme.dart';
import '../../widgets/alert_viewer.dart';

class AlertsSection extends StatefulWidget {
  const AlertsSection({super.key, required this.state});
  final AppState state;

  @override
  State<AlertsSection> createState() => _AlertsSectionState();
}

class _AlertsSectionState extends State<AlertsSection> {
  bool _alarmsOnly = false;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.state.alerts,
      builder: (context, _) {
        final log = widget.state.alerts;
        final all = log.entries;
        final items = _alarmsOnly
            ? all.where((e) => e.severity != AlertSeverity.info).toList()
            : all;
        final pending = log.unackedCount;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(12, 6, 8, 6),
              color: IbvapColors.surfaceAlt,
              child: Row(
                children: [
                  _toggle('ALL', !_alarmsOnly, () => setState(() => _alarmsOnly = false)),
                  const SizedBox(width: 4),
                  _toggle('ALARMS', _alarmsOnly, () => setState(() => _alarmsOnly = true)),
                  const Spacer(),
                  if (pending > 0)
                    TextButton(
                      onPressed: log.acknowledgeAll,
                      style: TextButton.styleFrom(
                          foregroundColor: IbvapColors.red,
                          padding: const EdgeInsets.symmetric(horizontal: 8)),
                      child: Text('ACK ALL ($pending)',
                          style: IbvapText.label(size: 10, color: IbvapColors.red)),
                    ),
                ],
              ),
            ),
            // Fixed height, not just a cap: a max-height-only box shrinks to
            // fit while there are few entries, so every new alert reflows the
            // whole sidebar underneath it — Virtual Fences' Draw/Save/Cancel
            // buttons become a moving target the operator can click right past.
            SizedBox(
              height: 280,
              child: items.isEmpty
                  ? Center(
                      child: Text(_alarmsOnly ? 'No alarms.' : 'No alerts yet.',
                          style: const TextStyle(color: IbvapColors.muted, fontSize: 12)),
                    )
                  : ListView.separated(
                      padding: EdgeInsets.zero,
                      itemCount: items.length,
                      separatorBuilder: (_, _) => const Divider(height: 1),
                      itemBuilder: (context, i) => _row(context, items[i]),
                    ),
            ),
          ],
        );
      },
    );
  }

  Widget _toggle(String label, bool on, VoidCallback onTap) => InkWell(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
          decoration: BoxDecoration(
            color: on ? IbvapColors.tint(IbvapColors.green, 0.16) : Colors.transparent,
            border: Border.all(color: on ? IbvapColors.green : IbvapColors.border),
            borderRadius: BorderRadius.circular(kRadius),
          ),
          child: Text(label,
              style: IbvapText.label(size: 9.5, color: on ? IbvapColors.green : IbvapColors.muted)),
        ),
      );

  Widget _row(BuildContext context, AlertEntry e) {
    final log = widget.state.alerts;
    final pending = e.needsAck;
    return Container(
      decoration: BoxDecoration(
        color: pending ? IbvapColors.tint(IbvapColors.red, 0.10) : null,
        border: Border(
          left: BorderSide(
            color: e.severity == AlertSeverity.info
                ? Colors.transparent
                : e.color,
            width: 3,
          ),
        ),
      ),
      padding: const EdgeInsets.fromLTRB(10, 7, 10, 7),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(e.icon, size: 13, color: e.color),
              const SizedBox(width: 6),
              Text(e.cam, style: IbvapText.data(size: 11, weight: FontWeight.w700)),
              const SizedBox(width: 7),
              Flexible(
                child: Text(e.title,
                    overflow: TextOverflow.ellipsis,
                    style: IbvapText.label(size: 10, color: e.color)),
              ),
              const Spacer(),
              if (pending)
                InkWell(
                  onTap: () => log.acknowledge(e),
                  child: Container(
                    margin: const EdgeInsets.only(right: 8),
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                    decoration: BoxDecoration(
                      color: IbvapColors.red,
                      borderRadius: BorderRadius.circular(kRadius),
                    ),
                    child: Text('ACK', style: IbvapText.label(size: 9, color: Colors.white)),
                  ),
                )
              else if (e.ackedAt != null)
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: Tooltip(
                    message: 'Acknowledged ${_hms(e.ackedAt!)}',
                    child: const Icon(Icons.check, size: 12, color: IbvapColors.faint),
                  ),
                ),
              Text(_hms(e.at), style: IbvapText.data(size: 10, color: IbvapColors.muted)),
            ],
          ),
          if (e.detail.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 2, left: 19),
              child: Text(e.detail,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(color: IbvapColors.muted, fontSize: 11)),
            ),
          if (e.thumbFile != null)
            Padding(
              padding: const EdgeInsets.only(top: 6, left: 19),
              child: GestureDetector(
                onTap: () => AlertViewer.show(context, widget.state, e),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(kRadius),
                  child: Image.network(
                    AlertViewer.thumbUrl(widget.state, e).toString(),
                    height: 88,
                    width: 156,
                    fit: BoxFit.cover,
                    gaplessPlayback: true,
                    cacheWidth: 312,
                    errorBuilder: (_, _, _) => Container(
                      height: 88,
                      width: 156,
                      color: Colors.black,
                      alignment: Alignment.center,
                      child: const Icon(Icons.broken_image_outlined,
                          color: IbvapColors.muted, size: 16),
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  static String _hms(DateTime d) =>
      '${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}:${d.second.toString().padLeft(2, '0')}';
}
