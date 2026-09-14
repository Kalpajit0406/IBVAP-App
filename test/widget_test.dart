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
import 'package:ibvap_app/services/tunnel_manager.dart';
import 'package:ibvap_app/state/alert_log.dart';
import 'package:ibvap_app/state/app_state.dart';
import 'package:ibvap_app/theme.dart';
import 'package:ibvap_app/util/describe_error.dart';

const _labels = [
  'MONITOR', 'OVERVIEW', 'CAMERAS', 'MODELS', 'SNAPSHOTS', 'ANPR', 'EVIDENCE', 'SETTINGS'
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
