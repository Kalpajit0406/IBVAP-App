import 'package:flutter/material.dart';

import '../../state/app_state.dart';
import '../../theme.dart';
import '../../widgets/backend_offline_card.dart';
import '../../widgets/model_switcher.dart';
import '../../widgets/panel.dart';
import '../../util/describe_error.dart';

class ModelsPanel extends StatefulWidget {
  const ModelsPanel({super.key, required this.state});

  final AppState state;

  @override
  State<ModelsPanel> createState() => _ModelsPanelState();
}

class _ModelsPanelState extends State<ModelsPanel> {
  bool _loading = false;
  String _selectedTrainMode = 'detector';
  bool _dryRun = false;
  static const int _epochs = 40;
  static const int _batch = 16;
  static const int _freeze = 10;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    if (_loading) return;
    setState(() => _loading = true);
    try {
      await widget.state.refreshNow();
    } catch (_) {}
    if (mounted) setState(() => _loading = false);
  }

  Future<void> _switchWeights(String weightsPath, String label) async {
    try {
      await widget.state.client.switchModel({
        'weights': weightsPath,
        'label': label,
        'name': label,
      });
      await widget.state.refreshNow();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Switched to $label')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to switch model: ${describeError(e)}')),
        );
      }
    }
  }

  Future<void> _switchRef(String refName) async {
    try {
      await widget.state.client.switchModel({'weights_ref': refName});
      await widget.state.refreshNow();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Switched to $refName')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to switch model: ${describeError(e)}')),
        );
      }
    }
  }

  Future<void> _startTraining() async {
    final modeLabel = switch (_selectedTrainMode) {
      'thermal' => 'Night / Thermal Model (LLVIP + FLIR)',
      'weapon' => 'Weapon Detector (YouTube-GDD)',
      'anpr' => 'ANPR Plate Detector',
      'posture' => 'Posture Anomaly Classifier',
      _ => 'Continuous Learning Detector',
    };

    final confirm = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: const Color(0xFF1E293B),
        title: Text('Start Training: $modeLabel?'),
        content: Text(
          _dryRun
              ? 'Run a DRY-RUN test to assemble and validate datasets without starting PyTorch training.'
              : 'Start fine-tuning with $_epochs epochs and batch size $_batch on CUDA GPU.',
          style: const TextStyle(color: IbvapColors.muted, fontSize: 13),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: IbvapColors.green, foregroundColor: Colors.black),
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Start Training'),
          ),
        ],
      ),
    );

    if (confirm != true) return;

    try {
      final res = await widget.state.client.learnRetrain(
        mode: _selectedTrainMode,
        dryRun: _dryRun,
        epochs: _epochs,
        batch: _batch,
        freeze: _freeze,
      );
      await widget.state.refreshNow();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Training started: ${res["script"] ?? _selectedTrainMode}')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to start training: ${describeError(e)}')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    final isOnline = s.link == LinkState.online;

    if (!isOnline) {
      return ListView(
        padding: const EdgeInsets.all(16),
        children: [
          BackendOfflineCard(state: s),
        ],
      );
    }

    final modelsData = s.models;
    final activeModel = (modelsData['active'] ?? s.status?.activeModel ?? 'nano').toString();
    final switching = modelsData['switching'] == true || (s.status?.switching ?? false);

    final registryList = (modelsData['registry'] as List?)?.cast<Map<String, dynamic>>() ?? [];
    final weaponList = (modelsData['weapons'] as List?)?.cast<Map<String, dynamic>>() ?? [];
    final filesList = (modelsData['files'] as List?)?.cast<Map<String, dynamic>>() ?? [];
    final lastTrain = (s.status?.learning['last_train'] as Map?)?.cast<String, dynamic>() ?? {};
    final isTraining = lastTrain['state'] == 'running';

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // ── Active Model Selector ──────────────────────────────────────────
        Panel(
          title: 'Active Surveillance Model',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ModelSwitcher(state: s),
              const SizedBox(height: 12),
              Wrap(
                spacing: 16,
                runSpacing: 8,
                children: [
                  _statBadge('Active Model', activeModel.toUpperCase(), IbvapColors.green),
                  _statBadge('Status', switching ? 'HOT-SWAPPING…' : 'LIVE INFERENCE',
                      switching ? IbvapColors.orange : IbvapColors.green),
                  _statBadge('Inference Speed', '${s.status?.inferenceMs.toStringAsFixed(1) ?? "-"} ms/frame', null),
                  _statBadge('GPU Saving', '${s.status?.gpuSavingPct.toStringAsFixed(0) ?? "-"}%', IbvapColors.orange),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),

        // ── Continually Learned & Thermal Models ───────────────────────────
        Panel(
          title: 'Continually Learned & Thermal Models (${registryList.length})',
          child: registryList.isEmpty
              ? const Text('No custom fine-tuned models registered yet.',
                  style: TextStyle(color: IbvapColors.muted, fontSize: 12))
              : Column(
                  children: [
                    for (final item in registryList) _buildRegistryCard(item, activeModel),
                  ],
                ),
        ),
        const SizedBox(height: 16),

        // ── Specialized Task Models (Weapon, ANPR, Posture) ────────────────
        Panel(
          title: 'Specialized Sub-Models (Weapons, ANPR, Posture)',
          child: Column(
            children: [
              _buildWeaponModelCard(weaponList),
              const SizedBox(height: 10),
              _buildAnprModelCard(s),
              const SizedBox(height: 10),
              _buildPostureModelCard(s),
            ],
          ),
        ),
        const SizedBox(height: 16),

        // ── Model Weights & File Scanner ───────────────────────────────────
        Panel(
          title: 'All Model Files & Future Weights Scanner (${filesList.length} files in models/)',
          child: filesList.isEmpty
              ? const Text('No weight files found in models/ directory.',
                  style: TextStyle(color: IbvapColors.muted, fontSize: 12))
              : Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Every .pt, .engine, and .onnx weight file placed in the models/ folder is listed below. You can hot-swap the detector to any model live.',
                      style: TextStyle(color: IbvapColors.muted, fontSize: 12),
                    ),
                    const SizedBox(height: 12),
                    for (final file in filesList) _buildFileRow(file, activeModel),
                  ],
                ),
        ),
        const SizedBox(height: 16),

        // ── Training & Fine-Tuning Studio ──────────────────────────────────
        Panel(
          title: 'Model Training & Fine-Tuning Studio',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (isTraining) ...[
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: const Color(0x22F59E0B),
                    borderRadius: BorderRadius.circular(kRadius),
                    border: Border.all(color: IbvapColors.orange),
                  ),
                  child: Row(
                    children: [
                      const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2, color: IbvapColors.orange),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Training in Progress: ${lastTrain["mode"]?.toString().toUpperCase() ?? "MODEL"}',
                              style: const TextStyle(
                                color: IbvapColors.orange,
                                fontSize: 13,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            Text(
                              'Script: ${lastTrain["script"] ?? "-"}',
                              style: const TextStyle(color: IbvapColors.muted, fontSize: 11),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
              ],

              // Training Form
              Wrap(
                spacing: 16,
                runSpacing: 12,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  DropdownButton<String>(
                    value: _selectedTrainMode,
                    dropdownColor: const Color(0xFF1E293B),
                    style: const TextStyle(color: IbvapColors.text, fontSize: 13),
                    items: const [
                      DropdownMenuItem(value: 'detector', child: Text('Continuous Learning (Reviewed Pool)')),
                      DropdownMenuItem(value: 'thermal', child: Text('Thermal / Night Model (LLVIP + FLIR)')),
                      DropdownMenuItem(value: 'weapon', child: Text('Weapon Detector (YouTube-GDD)')),
                      DropdownMenuItem(value: 'anpr', child: Text('ANPR License Plate Detector')),
                      DropdownMenuItem(value: 'posture', child: Text('Posture Anomaly Classifier')),
                    ],
                    onChanged: isTraining ? null : (v) => setState(() => _selectedTrainMode = v ?? 'detector'),
                  ),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Checkbox(
                        value: _dryRun,
                        activeColor: IbvapColors.green,
                        checkColor: Colors.black,
                        onChanged: isTraining ? null : (v) => setState(() => _dryRun = v ?? false),
                      ),
                      const Text('Dry Run (Assemble Only)', style: TextStyle(color: IbvapColors.text, fontSize: 12)),
                    ],
                  ),
                  FilledButton.icon(
                    style: FilledButton.styleFrom(
                      backgroundColor: isTraining ? IbvapColors.orange : IbvapColors.green,
                      foregroundColor: Colors.black,
                    ),
                    icon: const Icon(Icons.rocket_launch_rounded, size: 16),
                    label: Text(isTraining ? 'Training Active…' : 'Start Training'),
                    onPressed: isTraining ? null : _startTraining,
                  ),
                ],
              ),

              // Training Logs Stream
              if ((lastTrain['log_tail'] as List?)?.isNotEmpty ?? false) ...[
                const SizedBox(height: 12),
                const Text('TRAINING OUTPUT',
                    style: TextStyle(color: IbvapColors.muted, fontSize: 10, letterSpacing: 1, fontWeight: FontWeight.bold)),
                const SizedBox(height: 6),
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: const Color(0xFF020617),
                    borderRadius: BorderRadius.circular(kRadius),
                    border: Border.all(color: IbvapColors.border),
                  ),
                  constraints: const BoxConstraints(maxHeight: 180),
                  child: ListView(
                    shrinkWrap: true,
                    children: [
                      for (final line in (lastTrain['log_tail'] as List).cast<String>())
                        Text(
                          line,
                          style: const TextStyle(
                            color: Color(0xFF94A3B8),
                            fontFamily: 'monospace',
                            fontSize: 11,
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }

  Widget _statBadge(String label, String value, Color? color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(kRadius),
        border: Border.all(color: IbvapColors.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('$label: ', style: const TextStyle(color: IbvapColors.muted, fontSize: 11)),
          Text(
            value,
            style: TextStyle(
              color: color ?? IbvapColors.text,
              fontSize: 11,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRegistryCard(Map<String, dynamic> item, String activeModel) {
    final name = (item['name'] ?? '').toString();
    final label = (item['label'] ?? name).toString();
    final delta = item['delta'];
    final baseMap = (item['base_mAP'] as Map?)?['mAP50_95'] ?? (item['base_mAP'] is num ? item['base_mAP'] : null);
    final ftMap = (item['ft_mAP'] as Map?)?['mAP50_95'] ?? (item['ft_mAP'] is num ? item['ft_mAP'] : null);
    final dataset = item['dataset']?.toString() ?? 'Custom Harvest';
    final isActive = name == activeModel;

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isActive ? const Color(0x1F22C55E) : const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(kRadius),
        border: Border.all(color: isActive ? IbvapColors.green : IbvapColors.border),
      ),
      child: Row(
        children: [
          Icon(
            item['kind'] == 'thermal' ? Icons.nightlight_round : Icons.psychology,
            color: isActive ? IbvapColors.green : IbvapColors.muted,
            size: 24,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      label,
                      style: TextStyle(
                        color: isActive ? IbvapColors.green : IbvapColors.text,
                        fontWeight: FontWeight.bold,
                        fontSize: 13,
                      ),
                    ),
                    if (delta is num) ...[
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                        decoration: BoxDecoration(
                          color: const Color(0x3322C55E),
                          borderRadius: BorderRadius.circular(kRadius),
                        ),
                        child: Text(
                          '${delta >= 0 ? '+' : ''}${(delta * 100).toStringAsFixed(1)} mAP',
                          style: const TextStyle(
                            color: IbvapColors.green,
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  'Dataset: $dataset  •  Base: ${baseMap != null ? (baseMap * 100).toStringAsFixed(1) : "-"}% → Fine-Tuned: ${ftMap != null ? (ftMap * 100).toStringAsFixed(1) : "-"}%',
                  style: const TextStyle(color: IbvapColors.muted, fontSize: 11),
                ),
              ],
            ),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: isActive ? const Color(0x3322C55E) : IbvapColors.green,
              foregroundColor: isActive ? IbvapColors.green : Colors.black,
              visualDensity: VisualDensity.compact,
            ),
            onPressed: isActive ? null : () => _switchRef(name),
            child: Text(isActive ? 'Active' : 'Hot-Swap'),
          ),
        ],
      ),
    );
  }

  Widget _buildWeaponModelCard(List<Map<String, dynamic>> weaponList) {
    final w = weaponList.isNotEmpty ? weaponList.first : null;
    final metrics = (w?['metrics'] as Map?)?.cast<String, dynamic>() ?? {};

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(kRadius),
        border: Border.all(color: IbvapColors.border),
      ),
      child: Row(
        children: [
          const Icon(Icons.shield_outlined, color: IbvapColors.red, size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Row(
                  children: [
                    Text(
                      'Dedicated Firearm / Gun Detector (models/weapon_detector.pt)',
                      style: TextStyle(color: IbvapColors.text, fontWeight: FontWeight.bold, fontSize: 13),
                    ),
                    SizedBox(width: 8),
                    Text('[ACTIVE IN PIPELINE]',
                        style: TextStyle(color: IbvapColors.green, fontSize: 10, fontWeight: FontWeight.bold)),
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  'Precision: ${((metrics["precision"] ?? 0.87) * 100).toStringAsFixed(1)}%  •  Recall: ${((metrics["recall"] ?? 0.65) * 100).toStringAsFixed(1)}%  •  mAP50: ${((metrics["mAP50"] ?? 0.75) * 100).toStringAsFixed(1)}%  •  Trained on: YouTube-GDD',
                  style: const TextStyle(color: IbvapColors.muted, fontSize: 11),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAnprModelCard(AppState s) {
    final anpr = s.status?.anpr ?? {};
    final enabled = anpr['enabled'] == true;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(kRadius),
        border: Border.all(color: IbvapColors.border),
      ),
      child: Row(
        children: [
          const Icon(Icons.credit_card_rounded, color: IbvapColors.orange, size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Text(
                      'ANPR Number Plate Model (models/license_plate_detector.pt)',
                      style: TextStyle(color: IbvapColors.text, fontWeight: FontWeight.bold, fontSize: 13),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      enabled ? '[ENABLED]' : '[DISABLED]',
                      style: TextStyle(
                        color: enabled ? IbvapColors.green : IbvapColors.muted,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 2),
                const Text(
                  'Vehicle Crop Detector + Threaded EasyOCR + Indian License Plate Syntax Normalization',
                  style: TextStyle(color: IbvapColors.muted, fontSize: 11),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPostureModelCard(AppState s) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(kRadius),
        border: Border.all(color: IbvapColors.border),
      ),
      child: const Row(
        children: [
          Icon(Icons.accessibility_new_rounded, color: IbvapColors.green, size: 24),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      'Pose & Posture Keypoint Detector (yolo26n-pose.pt / yolo26m-pose.pt)',
                      style: TextStyle(color: IbvapColors.text, fontWeight: FontWeight.bold, fontSize: 13),
                    ),
                    SizedBox(width: 8),
                    Text('[ACTIVE]',
                        style: TextStyle(color: IbvapColors.green, fontSize: 10, fontWeight: FontWeight.bold)),
                  ],
                ),
                SizedBox(height: 2),
                Text(
                  '17-Keypoint Body Tracking for Anomaly Detection: LYING, CROUCH, SCAN, ARMS_UP, AIM / WEAPON_READY',
                  style: TextStyle(color: IbvapColors.muted, fontSize: 11),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFileRow(Map<String, dynamic> file, String activeModel) {
    final name = file['name']?.toString() ?? '';
    final filename = file['filename']?.toString() ?? '';
    final path = file['file']?.toString() ?? '';
    final sizeMb = file['size_mb']?.toString() ?? '0';
    final type = file['type']?.toString().toUpperCase() ?? 'PT';
    final isActive = name == activeModel || filename == activeModel;

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: isActive ? const Color(0x1F22C55E) : const Color(0xFF0F172A),
        borderRadius: BorderRadius.circular(kRadius),
        border: Border.all(color: isActive ? IbvapColors.green : IbvapColors.border),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: type == 'ENGINE'
                  ? const Color(0x3322C55E)
                  : type == 'ONNX'
                      ? const Color(0x3338BDF8)
                      : const Color(0x33F59E0B),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              type,
              style: TextStyle(
                color: type == 'ENGINE'
                    ? IbvapColors.green
                    : type == 'ONNX'
                        ? const Color(0xFF38BDF8)
                        : IbvapColors.orange,
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  filename,
                  style: TextStyle(
                    color: isActive ? IbvapColors.green : IbvapColors.text,
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                  ),
                ),
                Text(
                  'Path: $path  •  Size: $sizeMb MB',
                  style: const TextStyle(color: IbvapColors.muted, fontSize: 10.5),
                ),
              ],
            ),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: isActive ? const Color(0x3322C55E) : const Color(0xFF334155),
              foregroundColor: isActive ? IbvapColors.green : IbvapColors.text,
              visualDensity: VisualDensity.compact,
            ),
            onPressed: isActive ? null : () => _switchWeights(path, name),
            child: Text(isActive ? 'Active' : 'Load Weights'),
          ),
        ],
      ),
    );
  }
}
