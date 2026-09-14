import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../theme.dart';

class LinkStyle {
  static (Color, String) of(LinkState link) => switch (link) {
        LinkState.online => (IbvapColors.green, 'LINK UP'),
        LinkState.degraded => (IbvapColors.orange, 'DEGRADED'),
        LinkState.connecting => (IbvapColors.orange, 'CONNECTING'),
        LinkState.offline => (IbvapColors.red, 'LINK DOWN'),
      };
}

class ConnectionBadge extends StatelessWidget {
  const ConnectionBadge({super.key, required this.link, this.latency});

  final LinkState link;
  final Duration? latency;

  @override
  Widget build(BuildContext context) {
    final (color, label) = LinkStyle.of(link);
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 7,
          height: 7,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
            boxShadow: link == LinkState.online
                ? [BoxShadow(color: color.withValues(alpha: 0.5), blurRadius: 6)]
                : null,
          ),
        ),
        const SizedBox(width: 7),
        Text(label, style: IbvapText.label(size: 10, color: color)),
        if (link == LinkState.online && latency != null) ...[
          const SizedBox(width: 6),
          Text('${latency!.inMilliseconds}ms',
              style: IbvapText.data(size: 10, color: IbvapColors.muted)),
        ],
      ],
    );
  }
}
