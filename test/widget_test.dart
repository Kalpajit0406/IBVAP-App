import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:ibvap_app/api/ibvap_client.dart';
import 'package:ibvap_app/config/app_config.dart';
import 'package:ibvap_app/models/alert.dart';
import 'package:ibvap_app/models/fence.dart';
import 'package:ibvap_app/models/server_status.dart';
import 'package:ibvap_app/models/threat.dart';
import 'package:ibvap_app/screens/home_screen.dart';
import 'package:ibvap_app/screens/monitor_screen.dart';
import 'package:ibvap_app/screens/panels/fences_section.dart';
import 'package:ibvap_app/services/tunnel_manager.dart';
import 'package:ibvap_app/state/alert_log.dart';
import 'package:ibvap_app/state/app_state.dart';
import 'package:ibvap_app/state/fence_editor.dart';
import 'package:ibvap_app/theme.dart';
import 'package:ibvap_app/util/describe_error.dart';

const _labels = [
  'MONITOR', 'OVERVIEW', 'CAMERAS', 'MODELS', 'SNAPSHOTS', 'ANPR', 'FACES', 'EVIDENCE', 'SETTINGS'
];

Map<String, dynamic> _status({
  Map<String, dynamic> meta = const {},
  List<int> ids = const [0],
}) =>
    {
      'device': 'cuda:0',
      'mosaic': {'ids': ids, 'mode': 'grid', 'canvas': [960, 540], 'tiles': []},
      'meta': meta,
      'devices': [],
    };

/// A fence as the server stores and returns it: the behaviour settings, the
/// server-owned created_at, and a field this client has never heard of.
Map<String, dynamic> _serverFence(String id, {int cam = 0}) => {
      'id': id,
      'cam_id': cam,
      'kind': 'polygon',
      'points': [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8]],
      'direction': 'both',
      'targets': ['any'],
      'label': 'wire $id',
      'enabled': true,
      'created_at': 1789000000.5,
      'severity': 'Medium',
      'loiter_after_s': 45.0,
      'armed_from': '22:00',
      'armed_to': '05:00',
      'inbound': '',
      'future_field': {'nested': [1, 2, 3]},
    };

/// A FenceEditor already loaded from a mock server, plus every body it POSTs.
Future<(FenceEditor, List<Map<String, dynamic>>)> _fenceEditor(
    List<Map<String, dynamic>> served) async {
  final posted = <Map<String, dynamic>>[];
  final mock = MockClient((req) async {
    if (req.method == 'POST' && req.url.path == '/api/fences') {
      posted.add(jsonDecode(req.body) as Map<String, dynamic>);
      return http.Response('{"saved": 1}', 202,
          headers: {'content-type': 'application/json'});
    }
    return http.Response(jsonEncode({'fences': served}), 200,
        headers: {'content-type': 'application/json'});
  });
  final ed = FenceEditor(IbvapClient(client: mock));
  await ed.load();
  return (ed, posted);
}

void main() {
  group('shell', () {
    testWidgets('renders brand, chrome and every nav destination', (tester) async {
      tester.view.physicalSize = const Size(1440, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);

      final state = AppState(); // no start() -> no polling
      addTearDown(state.dispose);

      await tester.pumpWidget(MaterialApp(
        theme: buildIbvapTheme(),
        home: HomeScreen(state: state),
      ));
      await tester.pump();

      expect(find.text('BORDER SURVEILLANCE C2'), findsOneWidget);
      expect(find.text('THREAT CONDITION'), findsOneWidget);
      // Never claims NOMINAL without a live picture.
      expect(find.text('UNKNOWN'), findsOneWidget);
      for (final label in _labels) {
        expect(find.text(label), findsOneWidget, reason: label);
      }
    });

    testWidgets('every nav destination renders without a layout exception',
        (tester) async {
      // Regression guard: panels with Panel(expand: true) + scrolling children
      // threw at runtime (not caught by `flutter analyze`) the first time each
      // tab was opened.
      tester.view.physicalSize = const Size(1440, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);

      final state = AppState();
      addTearDown(state.dispose);

      await tester.pumpWidget(MaterialApp(
        theme: buildIbvapTheme(),
        home: HomeScreen(state: state),
      ));
      await tester.pump();

      for (final label in [..._labels.skip(1), _labels.first]) {
        await tester.tap(find.text(label));
        await tester.pump();
        expect(tester.takeException(), isNull, reason: 'tapping "$label"');
      }
    });

    testWidgets('alarm banner appears for a live critical alert and ACK clears it',
        (tester) async {
      tester.view.physicalSize = const Size(1440, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);

      final state = AppState();
      addTearDown(state.dispose);
      await tester.pumpWidget(MaterialApp(
        theme: buildIbvapTheme(),
        home: HomeScreen(state: state),
      ));

      state.alerts.ingestStatus(ServerStatus(_status(meta: {
        '0': {'level': 'Normal', 'breach': true, 'breach_zones': ['north wire']},
      })));
      await tester.pump();

      expect(find.text('ALARM'), findsOneWidget);
      expect(find.text('north wire'), findsWidgets);

      await tester.tap(find.text('ACK').first);
      await tester.pump();
      expect(find.text('ALARM'), findsNothing);
      expect(state.alerts.unackedCount, 0);
    });
  });

  group('Monitor', () {
    testWidgets('Streams menu removes a camera after confirmation', (tester) async {
      tester.view.physicalSize = const Size(1440, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);

      final deleted = <String>[];
      final mock = MockClient((req) async {
        if (req.method == 'DELETE') {
          deleted.add(req.url.path);
          return http.Response('{"deleted": 1}', 202,
              headers: {'content-type': 'application/json'});
        }
        final s = _status(ids: [0, 1], meta: {'0': {'level': 'Normal'}, '1': {'level': 'High'}});
        s['devices'] = [
          {'cam_id': 0, 'label': 'Gate', 'connected': true},
          {'cam_id': 1, 'label': 'Ridge', 'connected': true},
        ];
        return http.Response(jsonEncode(s), 200,
            headers: {'content-type': 'application/json'});
      });
      final state = AppState(client: IbvapClient(client: mock));
      addTearDown(state.dispose);
      await tester.runAsync(state.refreshNow);

      await tester.pumpWidget(MaterialApp(
        theme: buildIbvapTheme(),
        home: Scaffold(body: MonitorScreen(state: state)),
      ));
      await tester.pump();

      await tester.tap(find.text('Streams'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('CAM-01 · Ridge'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Remove stream…'));
      await tester.pumpAndSettle();
      expect(find.text('REMOVE STREAM'), findsWidgets, reason: 'confirmation first');
      expect(deleted, isEmpty);

      await tester.tap(find.widgetWithText(FilledButton, 'REMOVE STREAM'));
      await tester.runAsync(() => Future.delayed(const Duration(milliseconds: 50)));
      await tester.pump();
      expect(deleted, ['/api/streams/1']);
      // The live MJPEG view keeps retrying against the mock; tear it down.
      await tester.pumpWidget(const SizedBox());
    });
  });

  group('AlertLog', () {
    test('status edges alarm once per rising edge', () {
      final log = AlertLog();
      final breach = ServerStatus(_status(meta: {'0': {'breach': true}}));
      log.ingestStatus(breach);
      log.ingestStatus(breach); // still breaching: no second alarm
      expect(log.unackedCount, 1);
      log.ingestStatus(ServerStatus(_status(meta: {'0': {'breach': false}})));
      log.ingestStatus(breach); // new crossing
      expect(log.unackedCount, 2);
    });

    test('people at High and an unconfirmed AIM posture warn, never alarm', () {
      final log = AlertLog();
      log.ingestStatus(ServerStatus(_status(meta: {
        '0': {'level': 'High', 'persons': 3, 'weapon': false, 'weapon_tier': 'posture'},
      })));
      expect(log.unackedCount, 0);
      expect(log.entries.map((e) => e.title), containsAll(['HIGH', 'AIM POSTURE']));
      expect(log.entries.every((e) => e.severity == AlertSeverity.warning), isTrue);
    });

    test('a burst-confirmed criminal match alarms once per rising edge', () {
      final log = AlertLog();
      final hit = ServerStatus(_status(
          meta: {'0': {'criminal_match': true, 'criminal_name': 'Alice'}}));
      log.ingestStatus(hit);
      log.ingestStatus(hit); // still matched: no second alarm
      expect(log.unackedCount, 1);
      expect(log.entries.first.kind, AlertKind.face);
      expect(log.entries.first.title, 'CRIMINAL SPOTTED');
      expect(log.entries.first.detail, 'Alice');
      log.ingestStatus(ServerStatus(_status(meta: {'0': {'criminal_match': false}})));
      log.ingestStatus(hit); // new crossing
      expect(log.unackedCount, 2);
    });

    test('face results poll never alarms — the live edge above already did', () {
      final log = AlertLog();
      log.ingestFaceResults([
        {'ts': 1.0, 'cam_id': 0, 'track_id': 1, 'on_watchlist': true,
          'matched_name': 'Alice', 'files': ['0/a.jpg'], 'votes': 3, 'of': 5},
        {'ts': 2.0, 'cam_id': 0, 'track_id': 2, 'on_watchlist': false,
          'files': ['0/b.jpg'], 'votes': 0, 'of': 4},
      ]);
      expect(log.entries.length, 2);
      expect(log.entries.every((e) => e.kind == AlertKind.face), isTrue);
      expect(log.entries.every((e) => e.severity == AlertSeverity.info), isTrue);
      expect(log.unackedCount, 0);
      expect(log.entries.first.historical, isTrue); // first poll = history
    });

    test('first snapshot poll is history: logged, never alarmed', () {
      final log = AlertLog();
      log.ingestSnapshots([
        {'ts': 1.0, 'reason': 'posture', 'cam_id': 0, 'file': '0/a.jpg'},
      ]);
      expect(log.entries.length, 1);
      expect(log.entries.first.historical, isTrue);
      expect(log.unackedCount, 0);
    });

    test('de-dup survives overflow without re-flooding old records', () {
      final log = AlertLog();
      final page = [
        for (var i = 0; i < 20; i++)
          {'ts': i.toDouble(), 'reason': 'breach', 'cam_id': 0, 'track_id': i},
      ];
      log.ingestSnapshots(page);
      // Push well past the de-dup capacity with fresh records...
      for (var n = 0; n < 70; n++) {
        log.ingestSnapshots([
          for (var i = 0; i < 20; i++)
            {'ts': 1000.0 + n * 20 + i, 'reason': 'x', 'cam_id': 1, 'track_id': i},
        ]);
      }
      final before = log.entries.length;
      // ...then the server re-sends its newest page: nothing may be re-added.
      log.ingestSnapshots([
        for (var i = 0; i < 20; i++)
          {'ts': 1000.0 + 69 * 20 + i, 'reason': 'x', 'cam_id': 1, 'track_id': i},
      ]);
      expect(log.entries.length, before);
    });

    test('unacknowledged alarms are never evicted by the cap', () {
      final log = AlertLog();
      log.ingestStatus(ServerStatus(_status(meta: {'0': {'weapon': true}})));
      for (var n = 0; n < 40; n++) {
        log.ingestAnprResults([
          for (var i = 0; i < 20; i++) {'ts': n * 100.0 + i, 'cam_id': 2, 'track_id': i},
        ]);
      }
      expect(log.entries.length, lessThanOrEqualTo(301));
      expect(log.unackedCount, 1);
      expect(log.latestUnacked!.kind, AlertKind.weapon);
    });

    test('acknowledgeAll clears pending and records the time', () {
      final log = AlertLog();
      log.ingestStatus(ServerStatus(_status(
          ids: [0, 1], meta: {'0': {'level': 'Critical'}, '1': {'breach': true}})));
      expect(log.unackedCount, 2);
      log.acknowledgeAll();
      expect(log.unackedCount, 0);
      expect(log.entries.every((e) => e.severity != AlertSeverity.critical || e.ackedAt != null),
          isTrue);
    });
  });

  group('ThreatPicture', () {
    test('no link or no cameras is UNKNOWN, never NOMINAL', () {
      expect(ThreatPicture.from(null, linkUp: true).level, ThreatLevel.unknown);
      final s = ServerStatus(_status(meta: {'0': {'level': 'Normal'}}));
      expect(ThreatPicture.from(s, linkUp: false).level, ThreatLevel.unknown);
      expect(ThreatPicture.from(ServerStatus(_status(ids: [])), linkUp: true).level,
          ThreatLevel.unknown);
    });

    test('weapon/breach dominate and are listed first', () {
      final s = ServerStatus(_status(ids: [0, 1, 2], meta: {
        '0': {'level': 'High'},
        '1': {'level': 'Normal', 'weapon': true},
        '2': {'level': 'Normal'},
      }));
      final t = ThreatPicture.from(s, linkUp: true);
      expect(t.level, ThreatLevel.critical);
      expect(t.reasons.first, 'WEAPON CAM-01');
      expect(t.hotCams, [1, 0]);
    });

    test('all normal is NOMINAL', () {
      final s = ServerStatus(_status(ids: [0], meta: {'0': {'level': 'Normal'}}));
      expect(ThreatPicture.from(s, linkUp: true).level, ThreatLevel.nominal);
    });
  });

  group('AppState link hysteresis', () {
    test('one failed poll degrades, sustained failure goes offline', () async {
      var fail = false;
      final mock = MockClient((req) async {
        if (fail) throw http.ClientException('connection refused');
        return http.Response(jsonEncode(_status()), 200,
            headers: {'content-type': 'application/json'});
      });
      final state = AppState(client: IbvapClient(client: mock));
      addTearDown(state.dispose);

      await state.refreshNow();
      expect(state.link, LinkState.online);

      fail = true;
      await state.refreshNow();
      expect(state.link, LinkState.degraded, reason: 'a single miss must not drop the link');
      expect(state.status, isNotNull, reason: 'last good picture is kept');
      expect(state.threat.level, ThreatLevel.unknown,
          reason: 'but it is no longer asserted as current');

      await state.refreshNow();
      await state.refreshNow();
      // 3 failures but the last good poll was < offlineAfter ago: still degraded.
      expect(state.link, LinkState.degraded);

      fail = false;
      await state.refreshNow();
      expect(state.link, LinkState.online);
      expect(state.consecutiveFailures, 0);
    });

    test('never-connected server goes offline after repeated failures', () async {
      final mock = MockClient((_) async => throw http.ClientException('refused'));
      final state = AppState(client: IbvapClient(client: mock));
      addTearDown(state.dispose);
      await state.refreshNow();
      expect(state.link, LinkState.connecting);
      await state.refreshNow();
      await state.refreshNow();
      expect(state.link, LinkState.offline);
    });
  });

  group('config & errors', () {
    test('AppConfig default base URL', () {
      expect(AppConfig.defaultBaseUrl, 'http://127.0.0.1:8090');
    });

    test('validateBaseUrl accepts hosts and rejects junk', () {
      expect(AppConfig.validateBaseUrl('10.0.0.4:8090'), isNull);
      expect(AppConfig.validateBaseUrl('https://c2.local'), isNull);
      expect(AppConfig.validateBaseUrl(''), isNotNull);
      expect(AppConfig.validateBaseUrl('http://'), isNotNull);
      expect(AppConfig.validateBaseUrl('ftp://host'), isNotNull);
    });

    test('describeError turns auth and transport failures into guidance', () {
      expect(describeError(IbvapApiException('denied', statusCode: 401)),
          contains('API token'));
      expect(describeError(IbvapApiException('x', cause: http.ClientException('Connection refused'))),
          contains('unreachable'));
    });
  });

  group('models', () {
    test('MosaicLayout maps a canvas point to a camera tile', () {
      final layout = MosaicLayout.fromStatus({
        'mode': 'grid',
        'ids': [0, 1],
        'canvas': [960, 270],
        'tiles': [
          {'cam_id': 0, 'x': 0, 'y': 0, 'w': 480, 'h': 270},
          {'cam_id': 1, 'x': 480, 'y': 0, 'w': 480, 'h': 270},
        ],
      });
      expect(layout.tileAtCanvas(10, 10)?.camId, 0);
      expect(layout.tileAtCanvas(700, 100)?.camId, 1);
      expect(layout.tileAtCanvas(2000, 100), isNull);
    });

    test('Fence round-trips through JSON', () {
      final f = Fence(
        camId: 2,
        kind: 'line',
        points: [(0.1, 0.2), (0.5, 0.6)],
        direction: 'a2b',
        targets: ['person'],
        label: 'gate',
      );
      final back = Fence.fromJson(f.toJson());
      expect(back.camId, 2);
      expect(back.isLine, true);
      expect(back.points.length, 2);
      expect(back.direction, 'a2b');
      expect(back.targets, ['person']);
    });

    test('Fence round-trips the behaviour settings', () {
      final back = Fence.fromJson(Fence(
        camId: 0,
        kind: 'polygon',
        points: [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)],
        severity: 'Medium',
        loiterAfterS: 45,
        armedFrom: '22:00',
        armedTo: '05:30',
        inbound: 'b2a',
      ).toJson());
      expect(back.severity, 'Medium');
      expect(back.loiterAfterS, 45);
      expect(back.armedFrom, '22:00');
      expect(back.armedTo, '05:30');
      expect(back.inbound, 'b2a');
    });

    test('Fence keeps keys it does not manage (created_at, future fields)', () {
      // The console saves by re-POSTing the whole list, so any key this class
      // drops is erased from every fence on every save. created_at is
      // server-owned and was being reset each time for exactly this reason.
      final server = {
        'id': 'f_1', 'cam_id': 0, 'kind': 'polygon',
        'points': [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
        'created_at': 1789000000.5,
        'future_field': {'nested': [1, 2, 3]},
      };
      final out = Fence.fromJson(server).toJson();
      expect(out['created_at'], 1789000000.5);
      expect(out['future_field'], {'nested': [1, 2, 3]});
    });

    test('a managed key is never shadowed by a stale copy in extra', () {
      final f = Fence.fromJson({
        'cam_id': 0, 'kind': 'polygon',
        'points': [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
        'severity': 'Low',
      }).copyWith(severity: 'High', extra: {'severity': 'Info', 'x': 1});
      expect(f.toJson()['severity'], 'High');
      expect(f.toJson()['x'], 1);
    });

    test('a fence from before these fields existed loads with defaults', () {
      final f = Fence.fromJson({
        'cam_id': 0, 'kind': 'polygon',
        'points': [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
      });
      expect(f.severity, 'Critical');
      expect(f.loiterAfterS, 0);
      expect(f.armedFrom, '');
      expect(f.inbound, '');
      expect(f.extra, isEmpty);
    });
  });

  group('FenceEditor', () {
    // The regression this group exists for: saving from the console re-POSTs
    // the whole fence list, and used to strip every behaviour setting (and the
    // server's created_at) from all of it.
    test('toggling one fence does not erase settings on any fence', () async {
      final (ed, posted) = await _fenceEditor(
          [_serverFence('f_a'), _serverFence('f_b', cam: 1)]);
      expect(await ed.setEnabled('f_a', false), isNull);

      final sent = (posted.single['fences'] as List).cast<Map<String, dynamic>>();
      expect(sent.length, 2);
      for (final f in sent) {
        expect(f['severity'], 'Medium', reason: f['id']);
        expect(f['loiter_after_s'], 45.0, reason: f['id']);
        expect(f['armed_from'], '22:00', reason: f['id']);
        expect(f['armed_to'], '05:00', reason: f['id']);
        expect(f['created_at'], 1789000000.5, reason: f['id']);
        expect(f['future_field'], {'nested': [1, 2, 3]}, reason: f['id']);
      }
      expect(sent.firstWhere((f) => f['id'] == 'f_a')['enabled'], false);
      expect(sent.firstWhere((f) => f['id'] == 'f_b')['enabled'], true);
    });

    test('editing a fence saves the new settings and keeps the rest', () async {
      final (ed, posted) = await _fenceEditor(
          [_serverFence('f_a'), _serverFence('f_b', cam: 1)]);
      ed.startEdit('f_a');
      // The form is pre-filled from the stored fence, not blank.
      expect(ed.severity, 'Medium');
      expect(ed.loiterText, '45');
      expect(ed.armedFrom, '22:00');
      expect(ed.armedTo, '05:00');

      ed
        ..setLabel('north wire')
        ..setSeverity('High')
        ..setLoiterText('90')
        ..setArmedFrom('20:00')
        ..setArmedTo('06:30');
      expect(await ed.save(), isNull);

      final sent = (posted.single['fences'] as List).cast<Map<String, dynamic>>();
      final a = sent.firstWhere((f) => f['id'] == 'f_a');
      expect(a['label'], 'north wire');
      expect(a['severity'], 'High');
      expect(a['loiter_after_s'], 90.0);
      expect(a['armed_from'], '20:00');
      expect(a['armed_to'], '06:30');
      // Rebuilding the fence from the form used to discard these.
      expect(a['created_at'], 1789000000.5);
      expect(a['future_field'], {'nested': [1, 2, 3]});
      // and the fence that was not edited is untouched.
      final b = sent.firstWhere((f) => f['id'] == 'f_b');
      expect(b['severity'], 'Medium');
      expect(b['loiter_after_s'], 45.0);
    });

    test('a new fence is created with the settings chosen before Draw', () async {
      final (ed, posted) = await _fenceEditor([]);
      ed
        ..setSeverity('Low')
        ..setLoiterText('20')
        ..setArmedFrom('18:00')
        ..setArmedTo('06:00')
        ..setLabel('gate');
      ed.startDraw(0, 'polygon');
      // The form is filled in first and Draw pressed second; starting to draw
      // must not throw away what the operator already chose.
      expect(ed.severity, 'Low');
      expect(ed.loiterText, '20');
      expect(ed.label, 'gate');
      ed.draft
        ..add((0.1, 0.1))
        ..add((0.9, 0.1))
        ..add((0.5, 0.9));
      expect(await ed.save(), isNull);

      final f = (posted.single['fences'] as List).single as Map<String, dynamic>;
      expect(f['severity'], 'Low');
      expect(f['loiter_after_s'], 20.0);
      expect(f['armed_from'], '18:00');
      expect(f['armed_to'], '06:00');
      expect(f['label'], 'gate');
    });

    test('cancel clears the settings, and only cancel/edit bump the revision',
        () async {
      final (ed, _) = await _fenceEditor([_serverFence('f_a')]);
      ed
        ..setSeverity('Low')
        ..setLoiterText('20');
      final r0 = ed.revision;
      ed.startDraw(0, 'polygon');
      expect(ed.revision, r0, reason: 'a keystroke-neutral action must not reset the form');
      ed.cancel();
      expect(ed.revision, r0 + 1);
      expect(ed.severity, 'Critical');
      expect(ed.loiterText, '');
      ed.startEdit('f_a');
      expect(ed.revision, r0 + 2);
    });

    test('invalid settings block saving and say why', () async {
      final (ed, posted) = await _fenceEditor([]);
      ed.startDraw(0, 'polygon');
      ed.draft
        ..add((0.1, 0.1))
        ..add((0.9, 0.1))
        ..add((0.5, 0.9));
      expect(ed.canSave, isTrue);

      // A lone time is treated as "always armed" by the server, so a fence the
      // operator meant to be night-only would quietly stay live all day.
      ed.setArmedFrom('22:00');
      expect(ed.settingsError, contains('both'));
      expect(ed.canSave, isFalse);
      expect(await ed.save(), contains('both'));

      ed.setArmedTo('25:00');
      expect(ed.settingsError, contains('HH:MM'));
      ed.setArmedTo('05:00');
      expect(ed.settingsError, isNull);
      expect(ed.canSave, isTrue);

      ed.setLoiterText('abc');
      expect(ed.settingsError, contains('Loiter'));
      ed.setLoiterText('-5');
      expect(ed.settingsError, contains('Loiter'));
      ed.setLoiterText('');
      expect(ed.settingsError, isNull);
      expect(posted, isEmpty, reason: 'nothing may be sent while invalid');
    });

    testWidgets('the form shows the settings, follows an edit, and feeds the editor',
        (tester) async {
      tester.view.physicalSize = const Size(700, 1600);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);

      final mock = MockClient((req) async {
        final body = req.url.path == '/api/fences'
            ? {'fences': [_serverFence('f_a')]}
            : _status(ids: [0]);
        return http.Response(jsonEncode(body), 200,
            headers: {'content-type': 'application/json'});
      });
      final state = AppState(client: IbvapClient(client: mock));
      addTearDown(state.dispose);
      await tester.runAsync(state.refreshNow);
      await tester.runAsync(state.fences.load);

      await tester.pumpWidget(MaterialApp(
        theme: buildIbvapTheme(),
        home: Scaffold(
            body: SingleChildScrollView(child: FencesSection(state: state))),
      ));
      await tester.pump();
      expect(tester.takeException(), isNull);

      // The stored fence's settings are visible in the list.
      expect(find.textContaining('loiter 45s'), findsOneWidget);
      expect(find.textContaining('armed 22:00–05:00'), findsOneWidget);

      // Typing reaches the editor.
      final loiter = find.widgetWithText(TextField, 'Loiter alert after (seconds)');
      expect(loiter, findsOneWidget);
      await tester.enterText(loiter, '30');
      expect(state.fences.loiterText, '30');

      // Starting an edit loads that fence into the fields and dropdown.
      state.fences.startEdit('f_a');
      await tester.pump();
      expect(tester.widget<TextField>(loiter).controller!.text, '45');
      expect(find.text('Medium alert'), findsOneWidget,
          reason: 'the severity dropdown must follow the loaded fence');
      expect(tester.widget<TextField>(
              find.widgetWithText(TextField, 'Armed from')).controller!.text,
          '22:00');

      // A line shows the inbound choice instead of the loiter field.
      state.fences.cancel();
      state.fences.setKind('line');
      await tester.pump();
      expect(find.text('Inbound: not set'), findsOneWidget);
      expect(find.widgetWithText(TextField, 'Loiter alert after (seconds)'),
          findsNothing);
      expect(tester.takeException(), isNull);
    });
  });

  group('TunnelManager', () {
    test('defaults to a phones-only quick tunnel with no public URL', () {
      final t = TunnelManager();
      addTearDown(t.dispose);
      expect(t.status, TunnelStatus.stopped);
      expect(t.phoneUrl(0), isNull);
      expect(TunnelTarget.fromId('bogus'), TunnelTarget.phones);
      expect(TunnelTarget.fromId('console'), TunnelTarget.console);
    });

    test('fails with install guidance when cloudflared cannot be found', () async {
      await AppConfig.instance.setTunnel(
          cloudflaredPath: r'Z:\definitely\missing\cloudflared.exe', token: '', hostname: '');
      final t = TunnelManager();
      addTearDown(t.dispose);
      final bin = await TunnelManager.locateBinary();
      if (bin != null) return; // installed on this machine: covered by the live test instead
      final ok = await t.start();
      expect(ok, isFalse);
      expect(t.status, TunnelStatus.failed);
      expect(t.error, contains('winget install'));
    });
  });
}
