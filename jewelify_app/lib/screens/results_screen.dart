import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../models/jewelry_recommendation.dart';
import '../widgets/prediction_module.dart';
import 'dart:developer' as developer;

class ResultsScreen extends StatefulWidget {
  final Map<String, dynamic>? initialResult;

  const ResultsScreen({super.key, this.initialResult});

  @override
  _ResultsScreenState createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  List<Map<String, dynamic>> _predictions = [];
  bool _isLoading = true;
  String _errorMessage = "";
  String? _predictionId;
  final Map<String, bool> _feedbackSubmitted = {
    'prediction1': false,
    'prediction2': false,
  };
  String _loadingMessage = "Processing your prediction...";
  bool _isLongPolling = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _fetchResult();
  }

  Future<void> _fetchResult() async {
    if (!mounted) return;

    setState(() {
      _isLoading = true;
      _errorMessage = "";
      _loadingMessage = "Processing your prediction...";
      _isLongPolling = false;
    });

    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    final token = authProvider.token;

    if (token == null) {
      setState(() {
        _errorMessage = "Unauthorized: Please log in.";
        _isLoading = false;
      });
      Navigator.pushReplacementNamed(context, '/login');
      return;
    }

    final routeArgs =
        ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>?;
    final predictionData = widget.initialResult ?? routeArgs;
    final predictionId = predictionData?['prediction_id'] as String?;

    developer.log(
      'Fetching result with predictionId: $predictionId',
      name: 'ResultsScreen',
    );

    // Check for pre-fetched prediction data
    if (predictionId == null &&
        predictionData != null &&
        predictionData.containsKey('prediction1')) {
      setState(() {
        _predictionId = predictionData['prediction_id'];
        _predictions = [
          {
            'score':
                (predictionData['prediction1']['score'] is num
                    ? predictionData['prediction1']['score'].toDouble()
                    : 0.0),
            'category':
                predictionData['prediction1']['category']?.toString() ??
                'Not Assigned',
            'recommendations':
                (predictionData['prediction1']['recommendations']
                        as List<dynamic>?)
                    ?.map(
                      (rec) => JewelryRecommendation.fromJson(
                        rec as Map<String, dynamic>,
                      ),
                    )
                    .toList() ??
                [], // Keep as List<JewelryRecommendation>
            'face_image_path': predictionData['face_image_path'],
            'jewelry_image_path': predictionData['jewelry_image_path'],
            'model': 'prediction1',
            'feedback_required':
                predictionData['prediction1']['feedback_required'] ?? false,
            'overall_feedback':
                double.tryParse(
                  predictionData['prediction1']['overall_feedback']
                          ?.toString() ??
                      '0.5',
                ) ??
                0.5,
          },
          {
            'score':
                (predictionData['prediction2']['score'] is num
                    ? predictionData['prediction2']['score'].toDouble()
                    : 0.0),
            'category':
                predictionData['prediction2']['category']?.toString() ??
                'Not Assigned',
            'recommendations':
                (predictionData['prediction2']['recommendations']
                        as List<dynamic>?)
                    ?.map(
                      (rec) => JewelryRecommendation.fromJson(
                        rec as Map<String, dynamic>,
                      ),
                    )
                    .toList() ??
                [], // Keep as List<JewelryRecommendation>
            'face_image_path': predictionData['face_image_path'],
            'jewelry_image_path': predictionData['jewelry_image_path'],
            'model': 'prediction2',
            'feedback_required':
                predictionData['prediction2']['feedback_required'] ?? false,
            'overall_feedback':
                double.tryParse(
                  predictionData['prediction2']['overall_feedback']
                          ?.toString() ??
                      '0.5',
                ) ??
                0.5,
          },
        ];
        _isLoading = false;
      });
      developer.log(
        'Using pre-fetched prediction data: $_predictions',
        name: 'ResultsScreen',
      );

      // Check if the predictions are fallback values
      if (_isFallbackPrediction()) {
        setState(() {
          _errorMessage = "Prediction failed on the server. Please try again.";
          _predictions = [];
        });
      }
      return;
    }

    if (predictionId == null) {
      setState(() {
        _isLoading = false;
        _errorMessage = "No prediction data provided";
      });
      developer.log('No predictionId provided', name: 'ResultsScreen');
      return;
    }

    const maxAttempts = 30;
    const pollingInterval = Duration(seconds: 5);
    var attempts = 0;

    // Update loading message if polling takes longer than 60 seconds
    Future.delayed(const Duration(seconds: 60), () {
      if (mounted && _isLoading && !_isLongPolling) {
        setState(() {
          _isLongPolling = true;
          _loadingMessage = "Prediction is taking longer than expected...";
        });
      }
    });

    while (attempts < maxAttempts) {
      try {
        final apiUrl =
            'https://jewelify-server.onrender.com/predictions/get_prediction/$predictionId';
        developer.log(
          'Polling API URL: $apiUrl (Attempt ${attempts + 1})',
          name: 'ResultsScreen',
        );
        final response = await http
            .get(Uri.parse(apiUrl), headers: {'Authorization': 'Bearer $token'})
            .timeout(
              const Duration(seconds: 30),
              onTimeout: () {
                throw Exception(
                  "Request timed out while fetching prediction results",
                );
              },
            );

        developer.log(
          'Response status: ${response.statusCode}',
          name: 'ResultsScreen',
        );
        developer.log('Response body: ${response.body}', name: 'ResultsScreen');

        if (response.statusCode == 200) {
          final dynamic data = jsonDecode(response.body);
          if (data['validation_status'] == 'pending' ||
              data['prediction_status'] == 'pending') {
            attempts++;
            developer.log(
              'Prediction still pending, retrying... (Attempt ${attempts + 1})',
              name: 'ResultsScreen',
            );
            await Future.delayed(pollingInterval);
            continue;
          } else if (data['validation_status'] == 'failed') {
            setState(() {
              _isLoading = false;
              _errorMessage =
                  "Validation failed: Ensure a face is visible and jewelry is clear.";
            });
            return;
          } else if (data['prediction_status'] == 'failed') {
            setState(() {
              _isLoading = false;
              _errorMessage =
                  "Prediction failed: Server error during prediction.";
            });
            return;
          } else {
            setState(() {
              _predictionId = predictionId;
              _predictions = [
                {
                  'score':
                      (data['prediction1']['score'] is num
                          ? data['prediction1']['score'].toDouble()
                          : 0.0),
                  'category':
                      data['prediction1']['category']?.toString() ??
                      'Not Assigned',
                  'recommendations':
                      (data['prediction1']['recommendations'] as List<dynamic>?)
                          ?.map(
                            (rec) => JewelryRecommendation.fromJson(
                              rec as Map<String, dynamic>,
                            ),
                          )
                          .toList() ??
                      [], // Keep as List<JewelryRecommendation>
                  'face_image_path': data['face_image_path'],
                  'jewelry_image_path': data['jewelry_image_path'],
                  'model': 'prediction1',
                  'feedback_required':
                      data['prediction1']['feedback_required'] ?? false,
                  'overall_feedback':
                      double.tryParse(
                        data['prediction1']['overall_feedback']?.toString() ??
                            '0.5',
                      ) ??
                      0.5,
                },
                {
                  'score':
                      (data['prediction2']['score'] is num
                          ? data['prediction2']['score'].toDouble()
                          : 0.0),
                  'category':
                      data['prediction2']['category']?.toString() ??
                      'Not Assigned',
                  'recommendations':
                      (data['prediction2']['recommendations'] as List<dynamic>?)
                          ?.map(
                            (rec) => JewelryRecommendation.fromJson(
                              rec as Map<String, dynamic>,
                            ),
                          )
                          .toList() ??
                      [], // Keep as List<JewelryRecommendation>
                  'face_image_path': data['face_image_path'],
                  'jewelry_image_path': data['jewelry_image_path'],
                  'model': 'prediction2',
                  'feedback_required':
                      data['prediction2']['feedback_required'] ?? false,
                  'overall_feedback':
                      double.tryParse(
                        data['prediction2']['overall_feedback']?.toString() ??
                            '0.5',
                      ) ??
                      0.5,
                },
              ];
              _isLoading = false;
            });
            developer.log(
              'Prediction fetched successfully: $_predictions',
              name: 'ResultsScreen',
            );

            // Check if the predictions are fallback values
            if (_isFallbackPrediction()) {
              setState(() {
                _errorMessage =
                    "Prediction failed on the server. Please try again.";
                _predictions = [];
              });
            }
            return;
          }
        } else {
          final errorDetail =
              jsonDecode(response.body)['detail'] ?? 'Unknown error';
          setState(() {
            _isLoading = false;
            if (response.statusCode == 401) {
              _errorMessage =
                  "Unauthorized: Invalid or expired token. Please log in again.";
              Navigator.pushReplacementNamed(context, '/login');
            } else if (response.statusCode == 400 &&
                errorDetail == "Failed validation") {
              _errorMessage =
                  "Validation failed: Ensure a face is visible and jewelry is clear.";
            } else if (response.statusCode == 500 &&
                errorDetail == "Failed prediction") {
              _errorMessage =
                  "Prediction failed: Server error during prediction.";
            } else if (response.statusCode == 408) {
              _errorMessage =
                  "Request timed out waiting for prediction to complete.";
            } else {
              _errorMessage = "Failed to fetch result: $errorDetail";
            }
          });
          developer.log(
            'Fetch failed with status ${response.statusCode}: $errorDetail',
            name: 'ResultsScreen',
          );
          return;
        }
      } catch (e) {
        setState(() {
          _isLoading = false;
          if (e.toString().contains("timed out")) {
            _errorMessage =
                "Request timed out. Please check your connection and try again.";
          } else if (e.toString().contains("Connection failed") ||
              e.toString().contains("Network is unreachable")) {
            _errorMessage =
                "Network error: Unable to connect to the server. Please check your internet connection.";
          } else {
            _errorMessage = "Error fetching result: $e";
          }
        });
        developer.log('Fetch error: $e', name: 'ResultsScreen');
        return;
      }
    }

    setState(() {
      _isLoading = false;
      _errorMessage = "Prediction took too long to complete. Please try again.";
    });
    developer.log(
      'Prediction timed out after $maxAttempts attempts',
      name: 'ResultsScreen',
    );
  }

  bool _isFallbackPrediction() {
    // Check if the predictions are fallback values
    for (var prediction in _predictions) {
      if (prediction['category'] == 'Not Assigned' ||
          (prediction['recommendations'] as List).isEmpty ||
          prediction['score'] == 0.0) {
        return true;
      }
    }
    return false;
  }

  Future<void> _submitFeedback(
    String predictionId,
    String model, {
    String? recommendationName,
    String review = "yes",
  }) async {
    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    final token = authProvider.token;

    if (token == null) {
      setState(() {
        _errorMessage = "Unauthorized: Please log in.";
      });
      Navigator.pushReplacementNamed(context, '/login');
      return;
    }

    final feedbackType =
        recommendationName == null ? "prediction" : "recommendation";
    final apiUrl =
        'https://jewelify-server.onrender.com/predictions/feedback/$feedbackType';

    try {
      var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
      request.headers['Authorization'] = 'Bearer $token';
      request.fields['prediction_id'] = predictionId;
      request.fields['model_type'] = model;
      if (recommendationName != null) {
        request.fields['recommendation_name'] = recommendationName;
      }
      request.fields['score'] = review;

      developer.log(
        'Submitting feedback to $apiUrl with fields: ${request.fields}',
        name: 'ResultsScreen',
      );

      final response = await request.send().timeout(
        const Duration(seconds: 30),
        onTimeout: () {
          throw Exception("Request timed out while submitting feedback");
        },
      );
      final responseBody = await response.stream.bytesToString();

      developer.log(
        'Feedback response: ${response.statusCode} - $responseBody',
        name: 'ResultsScreen',
      );

      if (response.statusCode == 200) {
        if (recommendationName == null) {
          setState(() {
            _feedbackSubmitted[model] = true;
          });
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text("Feedback submitted successfully!")),
          );
        }
      } else if (response.statusCode == 401) {
        setState(() {
          _errorMessage =
              "Unauthorized: Invalid or expired token. Please log in again.";
        });
        Navigator.pushReplacementNamed(context, '/login');
      } else {
        developer.log(
          'Failed to submit feedback: ${response.statusCode} - $responseBody',
          name: 'ResultsScreen',
        );
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Failed to submit feedback. Please try again."),
          ),
        );
      }
    } catch (e) {
      developer.log('Error submitting feedback: $e', name: 'ResultsScreen');
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Error submitting feedback. Please try again."),
        ),
      );
    }
  }

  bool _canProceed() {
    return _feedbackSubmitted['prediction1'] == true &&
        _feedbackSubmitted['prediction2'] == true;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Your Result",
          style: TextStyle(fontWeight: FontWeight.w600),
        ),
        elevation: 0,
        backgroundColor: const Color(0xFFEDE7F6),
      ),
      body: Stack(
        children: [
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Color(0xFFEDE7F6), Color(0xFFEDE7F6)],
              ),
            ),
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(24.0),
                child:
                    _isLoading
                        ? Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const CircularProgressIndicator(),
                              const SizedBox(height: 16),
                              Text(_loadingMessage),
                            ],
                          ),
                        )
                        : _errorMessage.isNotEmpty
                        ? _buildErrorWidget(theme)
                        : _buildResultWidget(theme),
              ),
            ),
          ),
        ],
      ),
      bottomNavigationBar:
          _predictions.isNotEmpty && !_canProceed()
              ? Container(
                padding: const EdgeInsets.all(16.0),
                color: const Color(0xFFEDE7F6),
                child: const Text(
                  "Please provide feedback for both models to proceed.",
                  textAlign: TextAlign.center,
                ),
              )
              : null,
    );
  }

  Widget _buildResultWidget(ThemeData theme) {
    if (_predictions.isEmpty) {
      return const Center(child: Text("No prediction data available"));
    }

    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            itemCount: _predictions.length,
            itemBuilder: (context, index) {
              final prediction = _predictions[index];
              try {
                // Validate prediction data before passing to PredictionModule
                if (prediction['score'] == null ||
                    prediction['category'] == null ||
                    prediction['recommendations'] == null ||
                    prediction['face_image_path'] == null ||
                    prediction['jewelry_image_path'] == null ||
                    prediction['model'] == null ||
                    prediction['feedback_required'] == null ||
                    prediction['overall_feedback'] == null) {
                  developer.log(
                    'Invalid prediction data at index $index: $prediction',
                    name: 'ResultsScreen',
                  );
                  return const SizedBox.shrink(); // Return an empty widget to avoid rendering errors
                }

                return Builder(
                  builder: (context) {
                    try {
                      return PredictionModule(
                        modelName:
                            index == 0 ? "Model 1 (XGBoost)" : "Model 2 (MLP)",
                        prediction: prediction,
                        predictionId: _predictionId ?? '',
                        onFeedbackSubmit: _submitFeedback,
                      );
                    } catch (e, stackTrace) {
                      developer.log(
                        'Error rendering PredictionModule at index $index: $e',
                        name: 'ResultsScreen',
                        error: e,
                        stackTrace: stackTrace,
                      );
                      return Container(
                        padding: const EdgeInsets.all(16),
                        color: Colors.red.withOpacity(0.1),
                        child: Text(
                          'Error rendering prediction: $e',
                          style: const TextStyle(color: Colors.red),
                        ),
                      );
                    }
                  },
                );
              } catch (e, stackTrace) {
                developer.log(
                  'Error processing prediction at index $index: $e',
                  name: 'ResultsScreen',
                  error: e,
                  stackTrace: stackTrace,
                );
                return Container(
                  padding: const EdgeInsets.all(16),
                  color: Colors.red.withOpacity(0.1),
                  child: Text(
                    'Error processing prediction: $e',
                    style: const TextStyle(color: Colors.red),
                  ),
                );
              }
            },
          ),
        ),
        const SizedBox(height: 16),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed:
                    _canProceed()
                        ? () =>
                            Navigator.pushReplacementNamed(context, '/upload')
                        : null,
                icon: const Icon(Icons.refresh),
                label: const Text("Try Again"),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  side: BorderSide(color: theme.colorScheme.primary),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: ElevatedButton.icon(
                onPressed:
                    _canProceed()
                        ? () => Navigator.popUntil(
                          context,
                          (route) => route.isFirst,
                        )
                        : null,
                icon: const Icon(Icons.home),
                label: const Text("Home"),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildErrorWidget(ThemeData theme) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.error_outline, size: 80, color: theme.colorScheme.error),
        const SizedBox(height: 24),
        Text(
          "Error Fetching Result",
          style: theme.textTheme.headlineMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 16),
        Text(
          _errorMessage,
          style: theme.textTheme.bodyLarge,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 24),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: _fetchResult,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 12,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: const Text("Retry"),
            ),
            const SizedBox(width: 16),
            OutlinedButton(
              onPressed:
                  () => Navigator.pushReplacementNamed(context, '/upload'),
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 12,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: const Text("Go Back"),
            ),
          ],
        ),
      ],
    );
  }
}
