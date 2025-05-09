import 'package:flutter/material.dart';

class AppTheme {
  // Colors from both files, prioritizing and consolidating
  static const Color primaryColor = Color(
    0xFF2D4356,
  ); // Dark blue from both files
  static const Color secondaryLight = Color(0xFF81D4FA); // From first file
  static const Color secondaryDark = Color(0xFFB3E5FC); // From first file
  static const Color backgroundLight = Color(0xFFF5F5F5); // From second file
  static const Color backgroundDark = Color(0xFF2E2E2E); // From first file
  static const Color surfaceLight = Color(0xFFFFFFFF); // From first file
  static const Color surfaceDark = Color(0xFF3A3A3A); // From first file
  static const Color errorLight = Color(0xFFE57373); // From first file
  static const Color errorDark = Color(0xFFEF9A9A); // From first file
  static const Color accentColor = Color(
    0xFF333333,
  ); // From second file (used for text)
  static const Color dividerColor = Color(0xFFEEEEEE); // From second file
  static const Color inputBorderColor = Color(
    0xFFE0E0E0,
  ); // Renamed for clarity
  static const Color focusedBorderColor = Color(
    0xFFBDBDBD,
  ); // Renamed for clarity

  // Text colors
  static const Color textPrimaryLight = Color(0xFF212121); // From first file
  static const Color textSecondaryLight = Color(0xFF757575); // From first file
  static const Color textPrimaryDark = Color(0xFFE0E0E0); // From first file
  static const Color textSecondaryDark = Color(0xFFB0B0B0); // From first file

  // Text styles (combining both files)
  static const TextStyle titleStyle = TextStyle(
    fontSize: 22.0,
    fontWeight: FontWeight.bold,
    color: accentColor, // From second file
    fontFamily: 'Poppins', // Default to Poppins
  );

  static const TextStyle categoryStyle = TextStyle(
    fontSize: 18.0,
    fontWeight: FontWeight.w600,
    color: accentColor, // From second file
    fontFamily: 'Poppins', // Default to Poppins
  );

  static const TextStyle itemTitleStyle = TextStyle(
    fontSize: 16.0,
    fontWeight: FontWeight.w500,
    color: accentColor, // From second file
    fontFamily: 'Poppins', // Default to Poppins
  );

  static const TextStyle scoreStyle = TextStyle(
    fontSize: 14.0,
    fontWeight: FontWeight.w500,
    color: primaryColor, // From second file
    fontFamily: 'Poppins', // Default to Poppins
  );

  // Button styles (from second file)
  static final ButtonStyle primaryButtonStyle = ElevatedButton.styleFrom(
    backgroundColor: primaryColor,
    foregroundColor: Colors.white,
    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
  );

  static final ButtonStyle secondaryButtonStyle = OutlinedButton.styleFrom(
    foregroundColor: primaryColor,
    side: const BorderSide(color: primaryColor),
    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
  );

  // Card decoration (new addition to fix the error)
  static final BoxDecoration cardDecoration = BoxDecoration(
    color: Colors.white,
    borderRadius: BorderRadius.circular(8),
    boxShadow: [
      BoxShadow(
        color: Colors.black.withOpacity(0.05),
        blurRadius: 4,
        offset: const Offset(0, 2),
      ),
    ],
  );

  // ThemeData for light and dark modes
  static ThemeData lightTheme = ThemeData(
    brightness: Brightness.light,
    useMaterial3: true,
    fontFamily: 'Poppins', // Default font for the app is Poppins
    textTheme: TextTheme(
      displayLarge: TextStyle(fontFamily: 'Poppins'),
      headlineMedium: TextStyle(fontFamily: 'Poppins'),
      titleLarge: TextStyle(fontFamily: 'Poppins'),
      titleMedium: TextStyle(fontFamily: 'Poppins'),
      bodyLarge: TextStyle(fontFamily: 'Poppins'),
      bodyMedium: TextStyle(fontFamily: 'Poppins'),
      labelLarge: TextStyle(fontFamily: 'Poppins'),
    ),
    colorScheme: ColorScheme(
      brightness: Brightness.light,
      primary: primaryColor, // From both files
      onPrimary: Colors.white,
      secondary: secondaryLight, // From first file
      onSecondary: Colors.white,
      error: errorLight, // From first file
      onError: Colors.white,
      surface: surfaceLight, // From first file
      onSurface: textPrimaryLight, // Explicitly set for clarity
    ),
    scaffoldBackgroundColor:
        backgroundLight, // From second file, aligned with first
    appBarTheme: const AppBarTheme(
      backgroundColor: Colors.white, // From first file
      foregroundColor: primaryColor, // From both files
      elevation: 0, // From first file
      centerTitle: true,
      titleTextStyle: TextStyle(
        color: textPrimaryLight, // From first file
        fontSize: 20,
        fontWeight: FontWeight.bold,
        fontFamily: 'Poppins', // Use Poppins
      ),
    ),
    cardTheme: CardTheme(
      elevation: 4, // Adjusted from first file
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ), // From first file, adjusted to match cardDecoration radius
      shadowColor: Colors.black12, // From first file
      color: surfaceLight, // From first file
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: primaryButtonStyle, // Reference the static style defined above
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: secondaryButtonStyle, // Reference the static style defined above
    ),
    inputDecorationTheme: InputDecorationTheme(
      border: UnderlineInputBorder(
        borderSide: BorderSide(color: inputBorderColor), // Renamed for clarity
      ),
      enabledBorder: UnderlineInputBorder(
        borderSide: BorderSide(color: inputBorderColor), // Renamed for clarity
      ),
      focusedBorder: UnderlineInputBorder(
        borderSide: BorderSide(
          color: focusedBorderColor,
        ), // Renamed for clarity
      ),
      hintStyle: TextStyle(
        color: textSecondaryLight.withOpacity(0.6),
        fontSize: 14,
        fontFamily: 'Poppins', // Use Poppins
      ),
    ),
    dividerTheme: const DividerThemeData(
      color: dividerColor, // From second file
      thickness: 1,
      space: 1,
    ),
  );

  static ThemeData darkTheme = ThemeData(
    brightness: Brightness.dark,
    useMaterial3: true,
    fontFamily: 'Poppins', // Default font for the app is Poppins
    textTheme: TextTheme(
      displayLarge: TextStyle(fontFamily: 'Poppins'),
      headlineMedium: TextStyle(fontFamily: 'Poppins'),
      titleLarge: TextStyle(fontFamily: 'Poppins'),
      titleMedium: TextStyle(fontFamily: 'Poppins'),
      bodyLarge: TextStyle(fontFamily: 'Poppins'),
      bodyMedium: TextStyle(fontFamily: 'Poppins'),
      labelLarge: TextStyle(fontFamily: 'Poppins'),
    ),
    colorScheme: ColorScheme(
      brightness: Brightness.dark,
      primary: primaryColor, // From both files
      onPrimary: Colors.black87,
      secondary: secondaryDark, // From first file
      onSecondary: Colors.black87,
      error: errorDark, // From first file
      onError: Colors.black,
      surface: surfaceDark, // From first file
      onSurface: textPrimaryDark, // Explicitly set for clarity
    ),
    scaffoldBackgroundColor: backgroundDark, // From first file
    appBarTheme: const AppBarTheme(
      backgroundColor: surfaceDark, // From first file
      foregroundColor: primaryColor, // From both files
      elevation: 0, // From first file
      centerTitle: true,
      titleTextStyle: TextStyle(
        color: textPrimaryDark, // From first file
        fontSize: 20,
        fontWeight: FontWeight.bold,
        fontFamily: 'Poppins', // Use Poppins
      ),
    ),
    cardTheme: CardTheme(
      color: surfaceDark, // From first file
      elevation: 6, // Adjusted from first file
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ), // From first file, adjusted to match cardDecoration radius
      shadowColor: Colors.black26, // From first file
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: primaryButtonStyle, // Reference the static style defined above
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: secondaryButtonStyle, // Reference the static style defined above
    ),
    inputDecorationTheme: InputDecorationTheme(
      border: UnderlineInputBorder(
        borderSide: BorderSide(color: inputBorderColor), // Renamed for clarity
      ),
      enabledBorder: UnderlineInputBorder(
        borderSide: BorderSide(color: inputBorderColor), // Renamed for clarity
      ),
      focusedBorder: UnderlineInputBorder(
        borderSide: BorderSide(
          color: focusedBorderColor,
        ), // Renamed for clarity
      ),
      hintStyle: TextStyle(
        color: textSecondaryDark.withOpacity(0.6),
        fontSize: 14,
        fontFamily: 'Poppins', // Use Poppins
      ),
    ),
    dividerTheme: const DividerThemeData(
      color: dividerColor, // From second file
      thickness: 1,
      space: 1,
    ),
  );
}
