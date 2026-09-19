import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../models/fence.dart';
import '../../state/app_state.dart';
import '../../theme.dart';

class FencesSection extends StatefulWidget {
  const FencesSection({super.key, required this.state});
  final AppState state;

  @override
  State<FencesSection> createState() => _FencesSectionState();
}

class _FencesSectionState extends State<FencesSection> {
  final _label = TextEditingController();
  final _loiter = TextEditingController();
  final _follow = TextEditingController();
  final _from = TextEditingController();
  final _to = TextEditingController();
  int? _cam;
  // Which draft the text fields currently show; see FenceEditor.revision.
  int _rev = -1;

  @override
  void dispose() {
    _label.dispose();
    _loiter.dispose();
    _follow.dispose();
    _from.dispose();
    _to.dispose();
    super.dispose();
  }

  Future<void> _snack(String? err) async {
    if (err != null && mounted) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text(err)));
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([widget.state, widget.state.fences]),
      builder: (context, _) {
        final ed = widget.state.fences;
        final ids = widget.state.status?.cameraIds ?? const <int>[];
        _cam ??= ed.draftCam ?? (ids.isNotEmpty ? ids.first : null);
        if (_cam != null && ids.isNotEmpty && !ids.contains(_cam)) {
          _cam = ids.first;
        }
        // Push the draft into the text fields only when a different draft was
        // loaded (edit started, or cancelled/saved). Never on a keystroke —
        // the fields are the source of truth while the operator is typing.
        if (_rev != ed.revision) {
          _rev = ed.revision;
          _label.text = ed.label;
          _loiter.text = ed.loiterText;
          _follow.text = ed.followText;
          _from.text = ed.armedFrom;
          _to.text = ed.armedTo;
        }
        final active = widget.state.status?.geofenceActive ?? const {};

        return Padding(
          padding: const EdgeInsets.fromLTRB(12, 10, 12, 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _dropdown<int?>(
                value: _cam,
                hint: 'pick a camera',
                items: [
                  for (final id in ids)
                    DropdownMenuItem(
                        value: id,
                        child: Text('CAM-${id.toString().padLeft(2, '0')}')),
                ],
                onChanged: ed.armed ? null : (v) => setState(() => _cam = v),
              ),
              const SizedBox(height: 6),
              Row(
                children: [
                  Expanded(
                      child: _pill('Zone', ed.kind == 'polygon',
                          () => ed.setKind('polygon'), ed.armed)),
                  const SizedBox(width: 4),
                  Expanded(
                      child: _pill('Line', ed.kind == 'line',
                          () => ed.setKind('line'), ed.armed)),
                ],
              ),
              const SizedBox(height: 6),
              _dropdown<String>(
                value: ed.targets,
                items: const [
                  DropdownMenuItem(value: 'any', child: Text('Any target')),
                  DropdownMenuItem(
                      value: 'person', child: Text('Person only')),
                  DropdownMenuItem(
                      value: 'vehicle', child: Text('Vehicle only')),
                ],
                onChanged: (v) => ed.setTargets(v ?? 'any'),
              ),
              if (ed.kind == 'line') ...[
                const SizedBox(height: 6),
                _dropdown<String>(
                  value: ed.direction,
                  items: const [
                    DropdownMenuItem(
                        value: 'both', child: Text('Cross either way')),
                    DropdownMenuItem(value: 'a2b', child: Text('Cross A → B')),
                    DropdownMenuItem(value: 'b2a', child: Text('Cross B → A')),
                  ],
                  onChanged: (v) => ed.setDirection(v ?? 'both'),
                ),
              ],
              const SizedBox(height: 6),
              TextField(
                controller: _label,
                style: const TextStyle(fontSize: 12, color: IbvapColors.text),
                decoration: const InputDecoration(hintText: 'label (optional)'),
                onChanged: ed.setLabel,
              ),
              const SizedBox(height: 6),
              _dropdown<String>(
                // A value the server holds that this build does not list must
                // not crash the dropdown; show Critical, the safe default.
                value: Fence.severities.contains(ed.severity)
                    ? ed.severity
                    : 'Critical',
                items: [
                  for (final s in Fence.severities)
                    DropdownMenuItem(value: s, child: Text('$s alert')),
                ],
                onChanged: (v) => ed.setSeverity(v ?? 'Critical'),
              ),
              if (ed.kind == 'polygon') ...[
                const SizedBox(height: 6),
                TextField(
                  controller: _loiter,
                  style: const TextStyle(fontSize: 12, color: IbvapColors.text),
                  keyboardType: TextInputType.number,
                  inputFormatters: [
                    FilteringTextInputFormatter.allow(RegExp(r'[0-9.]')),
                  ],
                  decoration: const InputDecoration(
                      labelText: 'Loiter alert after (seconds)',
                      hintText: 'blank = off'),
                  onChanged: ed.setLoiterText,
                ),
              ] else ...[
                const SizedBox(height: 6),
                _dropdown<String>(
                  value: const ['a2b', 'b2a'].contains(ed.inbound)
                      ? ed.inbound
                      : '',
                  items: const [
                    DropdownMenuItem(value: '', child: Text('Inbound: not set')),
                    DropdownMenuItem(
                        value: 'a2b', child: Text('Inbound = A → B')),
                    DropdownMenuItem(
                        value: 'b2a', child: Text('Inbound = B → A')),
                  ],
                  onChanged: (v) => ed.setInbound(v ?? ''),
                ),
                const SizedBox(height: 6),
                TextField(
                  controller: _follow,
                  style: const TextStyle(fontSize: 12, color: IbvapColors.text),
                  keyboardType: TextInputType.number,
                  inputFormatters: [
                    FilteringTextInputFormatter.allow(RegExp(r'[0-9.]')),
                  ],
                  decoration: const InputDecoration(
                      labelText: 'Flag close-following within (seconds)',
                      hintText: 'blank = off'),
                  onChanged: ed.setFollowText,
                ),
                const Padding(
                  padding: EdgeInsets.only(top: 3),
                  child: Text(
                      'Flags a second crosser right behind the first. It reports '
                      'a pattern, not a verdict — expect it to fire where people '
                      'legitimately pass in pairs.',
                      style: TextStyle(color: IbvapColors.muted, fontSize: 9)),
                ),
              ],
              const SizedBox(height: 6),
              Row(
                children: [
                  Expanded(child: _timeField(_from, 'Armed from', ed.setArmedFrom)),
                  const SizedBox(width: 6),
                  Expanded(child: _timeField(_to, 'Armed to', ed.setArmedTo)),
                ],
              ),
              const Padding(
                padding: EdgeInsets.only(top: 3),
                child: Text('Leave both blank to keep the fence armed always.',
                    style: TextStyle(color: IbvapColors.muted, fontSize: 9)),
              ),
              if (ed.settingsError != null)
                Padding(
                  padding: const EdgeInsets.only(top: 3),
                  child: Text(ed.settingsError!,
                      style: const TextStyle(
                          color: IbvapColors.red, fontSize: 10)),
                ),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: _btn(
                      ed.armed ? 'Drawing…' : 'Draw',
                      ed.armed || _cam == null
                          ? null
                          : () => ed.startDraw(_cam!, ed.kind),
                      accent: ed.armed,
                    ),
                  ),
                  const SizedBox(width: 5),
                  Expanded(
                    child: _btn('Clear cam', _cam == null || ed.busy
                        ? null
                        : () async {
                            final ok = await showDialog<bool>(
                              context: context,
                              builder: (_) => AlertDialog(
                                backgroundColor: IbvapColors.surface,
                                title: const Text('Clear fences'),
                                content: Text(
                                    'Remove all fences on CAM-${_cam.toString().padLeft(2, '0')}?'),
                                actions: [
                                  TextButton(
                                      onPressed: () =>
                                          Navigator.pop(context, false),
                                      child: const Text('Cancel')),
                                  TextButton(
                                      onPressed: () =>
                                          Navigator.pop(context, true),
                                      child: const Text('Clear')),
                                ],
                              ),
                            );
                            if (ok == true) _snack(await ed.clearCam(_cam!));
                          }),
                  ),
                ],
              ),
              if (ed.armed) ...[
                const SizedBox(height: 5),
                Row(
                  children: [
                    Expanded(
                      child: _btn(
                          ed.editingId != null ? 'Save edit' : 'Save',
                          ed.canSave && !ed.busy
                              ? () async => _snack(await ed.save())
                              : null),
                    ),
                    const SizedBox(width: 5),
                    Expanded(child: _btn('Cancel', ed.cancel)),
                  ],
                ),
              ],
              const SizedBox(height: 10),
              if (ed.error != null)
                Text(ed.error!,
                    style:
                        const TextStyle(color: IbvapColors.red, fontSize: 10)),
              _list(ed, active),
            ],
          ),
        );
      },
    );
  }

  Widget _list(dynamic ed, Map<String, dynamic> active) {
    final fences = ed.fences as List;
    if (fences.isEmpty) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 10),
        child: Text('No fences yet.',
            style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
      );
    }
    return Column(
      children: [
        for (final f in fences)
          _fenceRow(f, active['${f.camId}'] is Map &&
              (active['${f.camId}'] as Map)[f.id] == true, ed),
      ],
    );
  }

  Widget _fenceRow(dynamic f, bool hot, dynamic ed) {
    final npts = (f.points as List).length;
    final kindTxt = f.kind == 'line'
        ? (npts > 2 ? 'line · ${npts - 1} seg' : 'line')
        : 'zone';
    return Container(
      decoration: BoxDecoration(
        color: hot ? const Color(0x22EF4444) : null,
        border: const Border(bottom: BorderSide(color: IbvapColors.border)),
      ),
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 28,
            child: Checkbox(
              value: f.enabled as bool,
              visualDensity: VisualDensity.compact,
              onChanged: (v) async => _snack(
                  await ed.setEnabled(f.id as String, v ?? true)),
            ),
          ),
          Expanded(
            child: InkWell(
              onTap: ed.armed ? null : () => ed.startEdit(f.id as String),
              child: Text.rich(
                TextSpan(children: [
                  TextSpan(
                      text: 'CAM-${f.camId.toString().padLeft(2, '0')} ',
                      style: const TextStyle(
                          color: Color(0xFF94A3B8),
                          fontSize: 11,
                          fontWeight: FontWeight.w700)),
                  if ((f.label as String).isNotEmpty)
                    TextSpan(
                        text: '· ${f.label} ',
                        style: const TextStyle(
                            color: IbvapColors.text, fontSize: 11)),
                  TextSpan(
                      text: kindTxt,
                      style: const TextStyle(
                          color: IbvapColors.blue,
                          fontSize: 9,
                          fontWeight: FontWeight.w700)),
                  if (_settingsSummary(f).isNotEmpty)
                    TextSpan(
                        text: '  ${_settingsSummary(f)}',
                        style: const TextStyle(
                            color: IbvapColors.muted, fontSize: 9)),
                ]),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.close, size: 14),
            color: IbvapColors.muted,
            visualDensity: VisualDensity.compact,
            tooltip: 'Delete',
            onPressed: ed.busy
                ? null
                : () async => _snack(await ed.delete(f.id as String)),
          ),
        ],
      ),
    );
  }

  Widget _timeField(
      TextEditingController c, String label, ValueChanged<String> onChanged) {
    return TextField(
      controller: c,
      style: const TextStyle(fontSize: 12, color: IbvapColors.text),
      inputFormatters: [
        FilteringTextInputFormatter.allow(RegExp(r'[0-9:]')),
        LengthLimitingTextInputFormatter(5),
      ],
      decoration: InputDecoration(labelText: label, hintText: 'HH:MM'),
      onChanged: onChanged,
    );
  }

  /// The non-default behaviour settings of a fence as one short line, so what
  /// was saved is visible in the list rather than only inside the edit form.
  String _settingsSummary(dynamic f) {
    final parts = <String>[];
    if (f.severity != 'Critical') parts.add('${f.severity}');
    final loiter = f.loiterAfterS as double;
    if (loiter > 0) {
      parts.add('loiter ${loiter == loiter.roundToDouble() ? loiter.round() : loiter}s');
    }
    if ((f.armedFrom as String).isNotEmpty) {
      parts.add('armed ${f.armedFrom}–${f.armedTo}');
    }
    if (f.inbound == 'a2b') parts.add('inbound A→B');
    if (f.inbound == 'b2a') parts.add('inbound B→A');
    final follow = f.followWindowS as double;
    if (follow > 0) {
      parts.add('follow ${follow == follow.roundToDouble() ? follow.round() : follow}s');
    }
    return parts.join(' · ');
  }

  Widget _dropdown<T>({
    required T value,
    String? hint,
    required List<DropdownMenuItem<T>> items,
    required ValueChanged<T?>? onChanged,
  }) {
    return DropdownButtonFormField<T>(
      initialValue: value,
      isDense: true,
      hint: hint == null ? null : Text(hint),
      style: const TextStyle(fontSize: 12, color: IbvapColors.text),
      dropdownColor: IbvapColors.surfaceAlt,
      items: items,
      onChanged: onChanged,
    );
  }

  Widget _pill(String label, bool on, VoidCallback onTap, bool disabled) {
    return InkWell(
      onTap: disabled ? null : onTap,
      borderRadius: BorderRadius.circular(kRadius),
      child: Container(
        alignment: Alignment.center,
        padding: const EdgeInsets.symmetric(vertical: 6),
        decoration: BoxDecoration(
          color: on ? IbvapColors.green : IbvapColors.surfaceAlt,
          borderRadius: BorderRadius.circular(kRadius),
          border: Border.all(color: IbvapColors.border),
        ),
        child: Text(label,
            style: TextStyle(
                color: on ? Colors.black : IbvapColors.muted,
                fontSize: 11,
                fontWeight: FontWeight.w700)),
      ),
    );
  }

  Widget _btn(String label, VoidCallback? onTap, {bool accent = false}) {
    return SizedBox(
      height: 30,
      child: OutlinedButton(
        onPressed: onTap,
        style: OutlinedButton.styleFrom(
          padding: EdgeInsets.zero,
          foregroundColor: accent ? IbvapColors.orange : IbvapColors.text,
          side: BorderSide(
              color: accent ? IbvapColors.orange : IbvapColors.border),
        ),
        child: Text(label, style: const TextStyle(fontSize: 11)),
      ),
    );
  }
}
