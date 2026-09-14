import 'package:flutter/material.dart';

import '../theme.dart';

/// Standard framed section used by every screen panel.
///
/// Two layout modes, chosen by [expand]:
///  * `false` (default) — the panel shrink-wraps its `child`. This is what every
///    panel embedded as one item of a scrolling `ListView` needs — that ambient
///    context hands each item *unbounded* height, so the panel must size itself
///    to its content.
///  * `true` — the panel instead fills whatever bounded height its ambient
///    context gives it, with an `Expanded` content slot. Use this when `child`
///    itself needs to fill the remaining space (e.g. a scrolling
///    `GridView`/`ListView`) and the panel is the page's own full-height
///    content — mixing `Expanded` content into the shrink-wrapping mode throws
///    "RenderFlex children have non-zero flex but incoming height constraints
///    are unbounded".
class Panel extends StatelessWidget {
  const Panel({
    super.key,
    required this.title,
    required this.child,
    this.actions = const [],
    this.padding = const EdgeInsets.all(16),
    this.expand = false,
    this.accent,
  });

  final String title;
  final Widget child;
  final List<Widget> actions;
  final EdgeInsets padding;
  final bool expand;

  /// Optional status colour for the header tick (defaults to neutral).
  final Color? accent;

  Widget _header() {
    return Container(
      height: 36,
      padding: const EdgeInsets.fromLTRB(0, 0, 8, 0),
      decoration: const BoxDecoration(
        color: IbvapColors.surfaceAlt,
        border: Border(bottom: BorderSide(color: IbvapColors.border)),
      ),
      child: Row(
        children: [
          Container(width: 3, color: accent ?? IbvapColors.borderStrong),
          const SizedBox(width: 11),
          Flexible(
            child: Text(
              title.toUpperCase(),
              overflow: TextOverflow.ellipsis,
              style: IbvapText.label(size: 10.5, color: IbvapColors.text),
            ),
          ),
          const Spacer(),
          ...actions,
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // The panel surface is a Material (not a coloured DecoratedBox) so ink
    // effects inside it — InkWell taps, ListTile/switch highlights — paint on
    // it instead of being hidden behind an opaque box.
    final column = Column(
      mainAxisSize: expand ? MainAxisSize.max : MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _header(),
        if (expand)
          Expanded(child: Padding(padding: padding, child: child))
        else
          Padding(padding: padding, child: child),
      ],
    );
    return Material(
      color: IbvapColors.surface,
      clipBehavior: Clip.antiAlias,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kRadius),
        side: const BorderSide(color: IbvapColors.border),
      ),
      child: column,
    );
  }
}

/// A label/value readout. Values are monospaced so they don't jitter per poll.
class StatTile extends StatelessWidget {
  const StatTile(this.label, this.value, {super.key, this.color});

  final String label;
  final String value;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label.toUpperCase(), style: IbvapText.label(size: 9)),
        const SizedBox(height: 3),
        Text(
          value,
          style: IbvapText.data(
            size: 18,
            color: color ?? IbvapColors.text,
            weight: FontWeight.w700,
          ),
        ),
      ],
    );
  }
}

/// Placeholder body for empty states.
class ComingSoon extends StatelessWidget {
  const ComingSoon(this.note, {super.key});
  final String note;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.inbox_outlined, color: IbvapColors.muted, size: 28),
          const SizedBox(height: 10),
          Text(note,
              textAlign: TextAlign.center,
              style: const TextStyle(color: IbvapColors.muted, fontSize: 12)),
        ],
      ),
    );
  }
}

/// Header used by the sections in the Monitor sidebar.
class SectionHeader extends StatelessWidget {
  const SectionHeader(this.title, {super.key, this.trailing, this.pill});

  final String title;
  final Widget? trailing;
  final String? pill;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 32,
      padding: const EdgeInsets.fromLTRB(12, 0, 8, 0),
      color: IbvapColors.surfaceAlt,
      alignment: Alignment.centerLeft,
      child: Row(
        children: [
          Container(width: 2, height: 12, color: IbvapColors.green),
          const SizedBox(width: 8),
          Text(title.toUpperCase(), style: IbvapText.label(size: 10, color: IbvapColors.text)),
          if (pill != null) ...[
            const SizedBox(width: 7),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
              decoration: BoxDecoration(
                color: const Color(0x14FFFFFF),
                borderRadius: BorderRadius.circular(kRadius),
              ),
              child: Text(pill!, style: IbvapText.data(size: 9.5, weight: FontWeight.w700)),
            ),
          ],
          const Spacer(),
          ?trailing,
        ],
      ),
    );
  }
}
