import 'package:flutter/material.dart';

import '../../state/app_state.dart';
import '../../theme.dart';
import '../../util/describe_error.dart';

class LearningSection extends StatefulWidget {
  const LearningSection({super.key, required this.state});
  final AppState state;

  @override
  State<LearningSection> createState() => _LearningSectionState();
}

class _LearningSectionState extends State<LearningSection> {
  List<dynamic> _pool = const [];
  int _lastPending = -1;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    if (_loading) return;
    setState(() => _loading = true);
    try {
      final j = await widget.state.client.learnPool(status: 'pending');
      final items = (j['items'] as List?) ?? const [];
      if (mounted) setState(() => _pool = items.take(20).toList());
    } catch (_) {
      // ignore — the section just shows what it has
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _review(String id, String verdict) async {
    final before = _pool;
    setState(() => _pool = _pool.where((e) => (e as Map)['id'] != id).toList());
    try {
      await widget.state.client.learnReview(id: id, verdict: verdict);
    } catch (e) {
      // The removal was optimistic — put the item back so a rejected review
      // doesn't look recorded.
      if (!mounted) return;
      setState(() => _pool = before);
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Review not saved: ${describeError(e)}')));
    }
  }

  Future<void> _retrain(bool dry) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: IbvapColors.surface,
        title: Text(dry ? 'Assemble training set?' : 'Fine-tune a new model?'),
        content: Text(dry
            ? 'Builds the set from kept crops. No training.'
            : 'Fine-tunes yolo26n on the reviewed pool. Takes a few minutes.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          TextButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Run')),
        ],
      ),
    );
    if (ok != true) return;
    try {
      await widget.state.client.learnRetrain(mode: 'detector', dryRun: dry);
      await widget.state.refreshNow();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Retrain failed: ${describeError(e)}')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.state,
      builder: (context, _) {
        final L = widget.state.status?.learning ?? const {};
        if (L['enabled'] == false) {
          return const Padding(
            padding: EdgeInsets.all(16),
            child: Text('Continuous learning is disabled on the server.',
                style: TextStyle(color: IbvapColors.muted, fontSize: 11)),
          );
        }
        final pending = (L['pool_pending'] as num?)?.toInt() ?? 0;
        if (pending != _lastPending) {
          _lastPending = pending;
          WidgetsBinding.instance.addPostFrameCallback((_) => _load());
        }
        final lt = (L['last_train'] as Map?)?.cast<String, dynamic>() ?? const {};

        return Padding(
          padding: const EdgeInsets.fromLTRB(14, 8, 12, 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'harvested ${L['harvested_total'] ?? 0} · pending $pending · '
                'kept ${L['pool_kept'] ?? 0} · dropped ${L['pool_dropped'] ?? 0} · '
                '${L['rate_per_min'] ?? 0}/min',
                style: const TextStyle(color: IbvapColors.muted, fontSize: 10.5),
              ),
              const SizedBox(height: 8),
              if (_pool.isEmpty)
                Text(
                  _loading ? 'loading candidates…' : 'No candidates to review.',
                  style:
                      const TextStyle(color: IbvapColors.muted, fontSize: 11),
                )
              else
                SizedBox(
                  height: 92,
                  child: ListView.separated(
                    scrollDirection: Axis.horizontal,
                    itemCount: _pool.length,
                    separatorBuilder: (_, _) => const SizedBox(width: 6),
                    itemBuilder: (context, i) {
                      final it = (_pool[i] as Map).cast<String, dynamic>();
                      final id = it['id'].toString();
                      return _card(id);
                    },
                  ),
                ),
              const SizedBox(height: 10),
              Row(
                children: [
                  Expanded(
                    child: _btn('Build set (dry-run)',
                        widget.state.isRetraining ? null : () => _retrain(true)),
                  ),
                  const SizedBox(width: 5),
                  Expanded(
                    child: _btn('Retrain',
                        widget.state.isRetraining ? null : () => _retrain(false)),
                  ),
                ],
              ),
              if (lt.isNotEmpty) ...[
                const SizedBox(height: 6),
                Text(_trainMsg(lt),
                    style: const TextStyle(
                        color: IbvapColors.muted, fontSize: 10)),
              ],
            ],
          ),
        );
      },
    );
  }

  String _trainMsg(Map<String, dynamic> lt) {
    final st = lt['state'];
    if (st == 'running') {
      final tail = (lt['log_tail'] as List?)?.isNotEmpty == true
          ? (lt['log_tail'] as List).last
          : '';
      return 'training (${lt['mode']})…  $tail';
    }
    if (st == 'done' && lt['delta'] != null) {
      final d = (lt['delta'] as num) * 100;
      return 'last: ${lt['name'] ?? lt['mode']}  Δ ${d >= 0 ? '+' : ''}${d.toStringAsFixed(1)} mAP — switch from Overview';
    }
    if (st == 'done') return 'last: ${lt['mode']} done';
    if (st == 'failed') {
      return 'last: ${lt['mode']} failed (${lt['reason'] ?? lt['error'] ?? '?'})';
    }
    return '';
  }

  Widget _card(String id) {
    return SizedBox(
      width: 92,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: Image.network(
              widget.state.client.learnThumbUrl(id).toString(),
              height: 58,
              width: 92,
              fit: BoxFit.cover,
              gaplessPlayback: true,
              errorBuilder: (_, _, _) => Container(
                  height: 58, width: 92, color: Colors.black),
            ),
          ),
          const SizedBox(height: 2),
          Row(
            children: [
              _mini('K', IbvapColors.green, () => _review(id, 'keep')),
              _mini('D', IbvapColors.red, () => _review(id, 'drop')),
              _mini('B', IbvapColors.blue, () => _review(id, 'background')),
            ],
          ),
        ],
      ),
    );
  }

  Widget _mini(String t, Color c, VoidCallback onTap) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        child: Container(
          margin: const EdgeInsets.symmetric(horizontal: 1),
          padding: const EdgeInsets.symmetric(vertical: 2),
          alignment: Alignment.center,
          decoration: BoxDecoration(
            border: Border.all(color: IbvapColors.border),
            borderRadius: BorderRadius.circular(3),
          ),
          child: Text(t,
              style: TextStyle(
                  color: c, fontSize: 9, fontWeight: FontWeight.w700)),
        ),
      ),
    );
  }

  Widget _btn(String label, VoidCallback? onTap) {
    return SizedBox(
      height: 30,
      child: OutlinedButton(
        onPressed: onTap,
        style: OutlinedButton.styleFrom(
          padding: EdgeInsets.zero,
          foregroundColor: IbvapColors.text,
          side: const BorderSide(color: IbvapColors.border),
        ),
        child: Text(label, style: const TextStyle(fontSize: 10.5)),
      ),
    );
  }
}
