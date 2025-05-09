import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart' as http_parser;
import 'dart:io';
import 'package:provider/provider.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import '../providers/auth_provider.dart';
import '../screens/image_storage.dart';
import '../screens/results_screen.dart';
import 'dart:developer' as developer;

class ProcessingScreen extends StatefulWidget {
  final Map<String, dynamic>? arguments;

  const ProcessingScreen({super.key, this.arguments});

  @override
  _ProcessingScreenState createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  bool _isError = false;
  String _errorMessage = "";
  bool _isCancelled = false;
  String _currentStatus = "Validating images locally...";
  bool _isLongWait = false;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();
    _processImages();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  Future<http.StreamedResponse> _sendRequestWithRetry(
    http.MultipartRequest request,
  ) async {
    const maxRetries = 2;
    var retries = 0;

    // Check for long wait after 30 seconds
    Future.delayed(const Duration(seconds: 30), () {
      if (mounted && !_isCancelled && !_isError && !_isLongWait) {
        setState(() {
          _isLongWait = true;
          _currentStatus = "Server is waking up, please wait...";
        });
      }
    });

    while (retries < maxRetries) {
      try {
        final response = await request.send().timeout(
          const Duration(seconds: 60),
          onTimeout: () {
            throw Exception("Request timed out while initiating prediction");
          },
        );
        return response;
      } catch (e) {
        retries++;
        if (retries == maxRetries) {
          rethrow;
        }
        await Future.delayed(const Duration(seconds: 5));
        developer.log('Retrying request (attempt ${retries + 1})');
      }
    }
    throw Exception("Failed to send request after $maxRetries attempts");
  }

  Future<void> _processImages() async {
    if (_isCancelled || widget.arguments == null) {
      if (mounted) Navigator.pop(context);
      return;
    }

    final faceFile = widget.arguments!['face'] as File;
    final jewelryFile = widget.arguments!['jewelry'] as File;
    final token = Provider.of<AuthProvider>(context, listen: false).token;

    if (token == null) {
      setState(() {
        _isError = true;
        _errorMessage = "Unauthorized: Please log in.";
      });
      Navigator.pushReplacementNamed(context, '/login');
      return;
    }

    setState(() => _currentStatus = "Validating images locally...");
    if (!await _validateImages(faceFile, jewelryFile)) {
      setState(() {
        _isError = true;
        _errorMessage =
            "Invalid images: Ensure a face is visible and jewelry is clear.";
      });
      return;
    }

    setState(() => _currentStatus = "Uploading images...");

    const apiUrl = 'https://jewelify-server.onrender.com/predictions/predict';
    var request = http.MultipartRequest('POST', Uri.parse(apiUrl));

    try {
      request.headers['Authorization'] = 'Bearer $token';

      String getMimeType(String path) {
        final ext = path.split('.').last.toLowerCase();
        return ext == 'png' ? 'image/png' : 'image/jpeg';
      }

      final facePath = await ImageStorage.saveImage(
        faceFile,
        'face_${DateTime.now().millisecondsSinceEpoch}',
      );
      final jewelryPath = await ImageStorage.saveImage(
        jewelryFile,
        'jewelry_${DateTime.now().millisecondsSinceEpoch}',
      );

      if (facePath == null || jewelryPath == null) {
        throw Exception("Failed to save images locally");
      }

      request.files.add(
        await http.MultipartFile.fromPath(
          'face',
          faceFile.path,
          contentType: http_parser.MediaType.parse(getMimeType(faceFile.path)),
        ),
      );
      request.files.add(
        await http.MultipartFile.fromPath(
          'jewelry',
          jewelryFile.path,
          contentType: http_parser.MediaType.parse(
            getMimeType(jewelryFile.path),
          ),
        ),
      );
      request.fields['face_image_path'] = facePath.split('/').last;
      request.fields['jewelry_image_path'] = jewelryPath.split('/').last;

      // Enhanced logging for debugging
      developer.log('Sending request to $apiUrl', name: 'ProcessingScreen');
      developer.log(
        'Request headers: ${request.headers}',
        name: 'ProcessingScreen',
      );
      developer.log(
        'Request fields: ${request.fields}',
        name: 'ProcessingScreen',
      );
      developer.log(
        'Request files: ${request.files.map((file) => file.field).toList()}',
        name: 'ProcessingScreen',
      );

      setState(() => _currentStatus = "Initiating prediction tasks...");

      final response = await _sendRequestWithRetry(request);
      final responseBody = await response.stream.bytesToString();

      developer.log(
        'Response status: ${response.statusCode}',
        name: 'ProcessingScreen',
      );
      developer.log('Response body: $responseBody', name: 'ProcessingScreen');

      if (response.statusCode == 200) {
        try {
          final result = jsonDecode(responseBody);
          setState(() => _currentStatus = "Prediction tasks initiated!");
          developer.log(
            'Prediction tasks initiated with result: $result',
            name: 'ProcessingScreen',
          );
          try {
            if (mounted && !_isCancelled) {
              developer.log(
                'Navigating to ResultsScreen',
                name: 'ProcessingScreen',
              );
              Navigator.of(context).pushReplacement(
                MaterialPageRoute(
                  builder: (context) => ResultsScreen(initialResult: result),
                ),
              );
              developer.log('Navigation executed', name: 'ProcessingScreen');
            } else {
              developer.log(
                'Navigation skipped: mounted=$mounted, isCancelled=$_isCancelled',
                name: 'ProcessingScreen',
              );
            }
          } catch (e) {
            developer.log('Navigation error: $e', name: 'ProcessingScreen');
            setState(() {
              _isError = true;
              _errorMessage = "Navigation error: $e";
            });
          }
        } catch (e) {
          developer.log(
            'Error parsing response body: $e',
            name: 'ProcessingScreen',
          );
          setState(() {
            _isError = true;
            _errorMessage = "Failed to parse server response: $e";
          });
        }
      } else {
        setState(() {
          _isError = true;
          if (response.statusCode == 401) {
            _errorMessage =
                "Unauthorized: Invalid or expired token. Please log in again.";
            Navigator.pushReplacementNamed(context, '/login');
          } else if (response.statusCode == 404) {
            _errorMessage =
                "Server endpoint not found. Please check the backend configuration.";
          } else if (response.statusCode == 400 &&
              responseBody.contains("Failed validation")) {
            _errorMessage =
                "Validation failed: Ensure a face is visible and jewelry is clear.";
          } else if (response.statusCode == 500 &&
              responseBody.contains("Failed prediction")) {
            _errorMessage =
                "Prediction failed: Server error during prediction.";
          } else if (response.statusCode == 500) {
            _errorMessage =
                "Server error: An unexpected error occurred on the server.";
          } else {
            _errorMessage = "Error: ${response.statusCode} - $responseBody";
          }
        });
      }
    } catch (e) {
      developer.log('Error processing images: $e', name: 'ProcessingScreen');
      if (mounted) {
        setState(() {
          _isError = true;
          if (e.toString().contains("timed out")) {
            _errorMessage =
                "Request timed out. Please check your internet connection and try again.";
          } else if (e.toString().contains("Connection failed") ||
              e.toString().contains("Network is unreachable")) {
            _errorMessage =
                "Network error: Unable to connect to the server. Please check your internet connection.";
          } else {
            _errorMessage = "Failed to process: $e";
          }
        });
      }
    }
  }

  Future<bool> _validateImages(File faceFile, File jewelryFile) async {
    developer.log('Starting image validation', name: 'ProcessingScreen');

    // Validate face image using Google ML Kit
    final inputImage = InputImage.fromFile(faceFile);
    final faceDetector = GoogleMlKit.vision.faceDetector();
    final faces = await faceDetector.processImage(inputImage);
    await faceDetector.close();

    if (faces.isEmpty) {
      developer.log(
        'No faces detected in the face image',
        name: 'ProcessingScreen',
      );
      return false;
    }
    developer.log('Faces detected: ${faces.length}', name: 'ProcessingScreen');

    // Validate jewelry image by checking its size
    final jewelryBytes = await jewelryFile.readAsBytes();
    if (jewelryBytes.length < 1024) {
      developer.log(
        'Jewelry image is too small: ${jewelryBytes.length} bytes',
        name: 'ProcessingScreen',
      );
      return false;
    }
    developer.log(
      'Jewelry image size is valid: ${jewelryBytes.length} bytes',
      name: 'ProcessingScreen',
    );

    return true;
  }

  void _cancelProcessing() {
    setState(() {
      _isCancelled = true;
      _animationController.stop();
    });
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return WillPopScope(
      onWillPop: () async {
        _cancelProcessing();
        return false;
      },
      child: Scaffold(
        appBar: AppBar(
          title: const Text(
            "Processing",
            style: TextStyle(fontWeight: FontWeight.w600),
          ),
        ),
        body: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors:
                  theme.brightness == Brightness.dark
                      ? [theme.colorScheme.surface, theme.colorScheme.surface]
                      : [theme.colorScheme.surface, theme.colorScheme.surface],
            ),
          ),
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(32.0),
              child:
                  _isError
                      ? _buildErrorWidget(theme)
                      : _buildLoadingWidget(theme),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLoadingWidget(ThemeData theme) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        RotationTransition(
          turns: _animationController,
          child: Icon(
            Icons.diamond,
            size: 100,
            color: theme.colorScheme.primary,
          ),
        ),
        const SizedBox(height: 48),
        Text(
          "Analyzing Your Style...",
          style: theme.textTheme.headlineMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 16),
        Text(
          _currentStatus,
          style: theme.textTheme.bodyLarge?.copyWith(
            color: theme.colorScheme.onSurface.withOpacity(0.7),
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 32),
        OutlinedButton.icon(
          onPressed: _cancelProcessing,
          icon: const Icon(Icons.cancel),
          label: const Text("Cancel"),
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            side: BorderSide(color: theme.colorScheme.error),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildErrorWidget(ThemeData theme) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.error_outline, size: 100, color: theme.colorScheme.error),
        const SizedBox(height: 48),
        Text(
          "Processing Error",
          style: theme.textTheme.headlineMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 16),
        Text(
          _errorMessage,
          style: theme.textTheme.bodyLarge?.copyWith(
            color: theme.colorScheme.onSurface.withOpacity(0.7),
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 32),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              onPressed: () {
                setState(() {
                  _isError = false;
                  _isLongWait = false;
                  _currentStatus = "Validating images locally...";
                  _animationController.repeat();
                });
                _processImages();
              },
              icon: const Icon(Icons.refresh),
              label: const Text("Retry"),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 12,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
            const SizedBox(width: 16),
            OutlinedButton.icon(
              onPressed: _cancelProcessing,
              icon: const Icon(Icons.arrow_back),
              label: const Text("Go Back"),
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 12,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }
}
