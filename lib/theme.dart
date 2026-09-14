import 'package:flutter/material.dart';

/// IBVAP command-client palette — a low-glare tactical console scheme.
/// Colour carries meaning only: green nominal, amber caution, red critical,
/// cyan informational. Nothing decorative is saturated, so an alarm is the
/// brightest thing on screen.
class IbvapColors {
  static const bg = Color(0xFF06080B);
  static const surface = Color(0xFF0C1016);
  static const surfaceAlt = Color(0xFF111720);
  static const surfaceRaised = Color(0xFF161D28);
  static const border = Color(0xFF1C2430);
  static const borderStrong = Color(0xFF2A3445);
  static const text = Color(0xFFD5DCE6);
  static const muted = Color(0xFF6A7687);
  static const faint = Color(0xFF3D4757);

  static const green = Color(0xFF2ED47A); // nominal / live / OK
  static const orange = Color(0xFFF2A516); // caution / High / idle
  static const red = Color(0xFFFF4040); // critical / breach / offline
  static const blue = Color(0xFF3CC4E6); // informational / plate
  static const plate = Color(0xFFF5C518);

  static Color tint(Color c, [double a = 0.14]) => c.withValues(alpha: a);
}

/// Text styles for operator data. Numbers, IDs, timestamps and hashes are
/// monospaced so columns don't jitter as values change each poll.
class IbvapText {
  static const mono = 'Consolas';
  static const monoFallback = ['Cascadia Mono', 'Courier New', 'monospace'];

  static TextStyle data({
    double size = 12,
    Color color = IbvapColors.text,
    FontWeight weight = FontWeight.w500,
    double letterSpacing = 0,
  }) =>
      TextStyle(
        fontFamily: mono,
        fontFamilyFallback: monoFallback,
        fontSize: size,
        color: color,
        fontWeight: weight,
        letterSpacing: letterSpacing,
        fontFeatures: const [FontFeature.tabularFigures()],
      );

  /// Upper-case section / field labels.
  static TextStyle label({double size = 10, Color color = IbvapColors.muted}) =>
      TextStyle(
        fontSize: size,
        color: color,
        fontWeight: FontWeight.w700,
        letterSpacing: 1.2,
      );
}

const kRadius = 3.0;

ThemeData buildIbvapTheme() {
  const scheme = ColorScheme.dark(
    primary: IbvapColors.green,
    onPrimary: Colors.black,
    secondary: IbvapColors.blue,
    surface: IbvapColors.surface,
    onSurface: IbvapColors.text,
    error: IbvapColors.red,
    outline: IbvapColors.borderStrong,
  );

  final sharp = RoundedRectangleBorder(borderRadius: BorderRadius.circular(kRadius));

  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: IbvapColors.bg,
    fontFamily: 'Segoe UI',
    visualDensity: VisualDensity.compact,
    splashFactory: NoSplash.splashFactory,
    dividerColor: IbvapColors.border,
    dividerTheme: const DividerThemeData(
      color: IbvapColors.border,
      thickness: 1,
      space: 1,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: IbvapColors.surface,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: IbvapColors.surface,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kRadius),
        side: const BorderSide(color: IbvapColors.border),
      ),
      margin: EdgeInsets.zero,
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: IbvapColors.surface,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kRadius),
        side: const BorderSide(color: IbvapColors.borderStrong),
      ),
      titleTextStyle: const TextStyle(
          color: IbvapColors.text, fontSize: 15, fontWeight: FontWeight.w700),
      contentTextStyle: const TextStyle(color: IbvapColors.text, fontSize: 13),
    ),
    popupMenuTheme: PopupMenuThemeData(
      color: IbvapColors.surfaceRaised,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kRadius),
        side: const BorderSide(color: IbvapColors.borderStrong),
      ),
    ),
    snackBarTheme: SnackBarThemeData(
      backgroundColor: IbvapColors.surfaceRaised,
      contentTextStyle: const TextStyle(color: IbvapColors.text, fontSize: 12),
      behavior: SnackBarBehavior.floating,
      width: 520,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kRadius),
        side: const BorderSide(color: IbvapColors.borderStrong),
      ),
    ),
    tooltipTheme: TooltipThemeData(
      waitDuration: const Duration(milliseconds: 400),
      textStyle: const TextStyle(color: IbvapColors.text, fontSize: 11),
      decoration: BoxDecoration(
        color: IbvapColors.surfaceRaised,
        border: Border.all(color: IbvapColors.borderStrong),
        borderRadius: BorderRadius.circular(kRadius),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        shape: sharp,
        textStyle: const TextStyle(
            fontSize: 12, fontWeight: FontWeight.w700, letterSpacing: 0.6),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        shape: sharp,
        foregroundColor: IbvapColors.text,
        side: const BorderSide(color: IbvapColors.borderStrong),
        textStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        shape: sharp,
        textStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
      ),
    ),
    navigationRailTheme: const NavigationRailThemeData(
      backgroundColor: IbvapColors.surface,
      indicatorColor: Color(0x222ED47A),
      indicatorShape: RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(kRadius))),
      selectedIconTheme: IconThemeData(color: IbvapColors.green, size: 20),
      unselectedIconTheme: IconThemeData(color: IbvapColors.muted, size: 20),
      selectedLabelTextStyle: TextStyle(
        color: IbvapColors.text,
        fontSize: 9.5,
        fontWeight: FontWeight.w800,
        letterSpacing: 1,
      ),
      unselectedLabelTextStyle: TextStyle(
        color: IbvapColors.muted,
        fontSize: 9.5,
        fontWeight: FontWeight.w600,
        letterSpacing: 1,
      ),
    ),
    sliderTheme: const SliderThemeData(
      activeTrackColor: IbvapColors.green,
      inactiveTrackColor: IbvapColors.border,
      thumbColor: IbvapColors.green,
      trackHeight: 2,
    ),
    inputDecorationTheme: InputDecorationTheme(
      isDense: true,
      filled: true,
      fillColor: const Color(0xFF090D12),
      contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 9),
      hintStyle: const TextStyle(color: IbvapColors.faint, fontSize: 12),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(kRadius),
        borderSide: const BorderSide(color: IbvapColors.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(kRadius),
        borderSide: const BorderSide(color: IbvapColors.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(kRadius),
        borderSide: const BorderSide(color: IbvapColors.green),
      ),
      errorBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(kRadius),
        borderSide: const BorderSide(color: IbvapColors.red),
      ),
    ),
    scrollbarTheme: ScrollbarThemeData(
      thickness: WidgetStateProperty.all(6),
      radius: const Radius.circular(kRadius),
      thumbColor: WidgetStateProperty.all(IbvapColors.borderStrong),
    ),
    textTheme: const TextTheme(
      bodyMedium: TextStyle(color: IbvapColors.text, fontSize: 13),
      bodySmall: TextStyle(color: IbvapColors.muted, fontSize: 11),
      titleMedium: TextStyle(
        color: IbvapColors.text,
        fontSize: 13,
        fontWeight: FontWeight.w600,
      ),
      labelSmall: TextStyle(
        color: IbvapColors.muted,
        fontSize: 9,
        letterSpacing: 0.8,
        fontWeight: FontWeight.w700,
      ),
    ),
  );
}
