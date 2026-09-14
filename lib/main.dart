import 'package:flutter/material.dart';

import 'config/app_config.dart';
import 'screens/home_screen.dart';
import 'state/app_state.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await AppConfig.instance.load();
  runApp(const IbvapApp());
}

class IbvapApp extends StatefulWidget {
  const IbvapApp({super.key});

  @override
  State<IbvapApp> createState() => _IbvapAppState();
}

class _IbvapAppState extends State<IbvapApp> {
  final AppState _state = AppState()..start();

  @override
  void dispose() {
    _state.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IBVAP Command Client',
      debugShowCheckedModeBanner: false,
      theme: buildIbvapTheme(),
      home: HomeScreen(state: _state),
    );
  }
}
