import 'package:flutter/material.dart';
import 'package:jewelify/screens/temp_registration.dart';
import 'package:provider/provider.dart';
import 'providers/auth_provider.dart';
import 'screens/login_screen.dart';
import 'screens/home_screen.dart';
import 'screens/upload_screen.dart';
import 'screens/processing_screen.dart';
import 'screens/results_screen.dart';
import 'screens/history_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final authProvider = AuthProvider();
  await authProvider.loadToken();

  runApp(
    MultiProvider(
      providers: [ChangeNotifierProvider(create: (_) => authProvider)],
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool isDarkMode = false;

  void toggleTheme() {
    setState(() => isDarkMode = !isDarkMode);
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<AuthProvider>(
      builder: (context, authProvider, child) {
        return MaterialApp(
          title: 'Jewelry Match',
          theme: ThemeData.light().copyWith(
            textTheme: ThemeData.light().textTheme.apply(fontFamily: 'Poppins'),
          ),
          darkTheme: ThemeData.dark().copyWith(
            textTheme: ThemeData.dark().textTheme.apply(fontFamily: 'Poppins'),
          ),
          themeMode: isDarkMode ? ThemeMode.dark : ThemeMode.light,
          initialRoute: authProvider.isAuthenticated ? '/home' : '/register',
          routes: {
            '/register': (context) => const RegisterScreen(),
            '/login': (context) => const LoginScreen(),
            '/home':
                (context) => HomeScreen(
                  toggleTheme: toggleTheme,
                  isDarkMode: isDarkMode,
                ),
            '/upload': (context) => const UploadScreen(),
            '/processing':
                (context) => ProcessingScreen(
                  arguments:
                      ModalRoute.of(context)?.settings.arguments
                          as Map<String, dynamic>?,
                ),
            '/results': (context) => const ResultsScreen(),
            '/history': (context) => const HistoryScreen(),
          },
          debugShowCheckedModeBanner: false,
        );
      },
    );
  }
}
