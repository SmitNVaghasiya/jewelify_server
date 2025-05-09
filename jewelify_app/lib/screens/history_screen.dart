import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../models/jewelry_recommendation.dart';
import '../widgets/expandable_item.dart';
import '../widgets/prediction_module.dart';

class HistoryItem {
  final String id;
  final String? userId;
  final String date;
  final Map<String, dynamic> prediction1;
  final Map<String, dynamic> prediction2;
  final String? faceImagePath;
  final String? jewelryImagePath;
  final String validationStatus;
  final String predictionStatus;
  bool isExpanded;

  HistoryItem({
    required this.id,
    this.userId,
    required this.date,
    required this.prediction1,
    required this.prediction2,
    this.faceImagePath,
    this.jewelryImagePath,
    required this.validationStatus,
    required this.predictionStatus,
    this.isExpanded = false,
  });

  factory HistoryItem.fromJson(Map<String, dynamic> json) {
    return HistoryItem(
      id: json['id']?.toString() ?? '',
      userId: json['user_id']?.toString(),
      date: json['timestamp']?.toString() ?? 'Unknown Date',
      prediction1: {
        'score':
            (json['prediction1']['score'] is num
                ? json['prediction1']['score'].toDouble()
                : 0.0),
        'category':
            json['prediction1']['category']?.toString() ?? 'Not Assigned',
        'recommendations':
            (json['prediction1']['recommendations'] as List<dynamic>?)
                ?.map(
                  (rec) => JewelryRecommendation.fromJson(
                    rec as Map<String, dynamic>,
                  ),
                )
                .toList() ??
            [],
        'overall_feedback':
            json['prediction1']['overall_feedback']?.toString() ??
            'Not Provided',
        'feedback_required':
            json['prediction1']['feedback_required']?.toString(),
        'face_image_path': json['face_image_path']?.toString(),
        'jewelry_image_path': json['jewelry_image_path']?.toString(),
        'model': 'prediction1',
      },
      prediction2: {
        'score':
            (json['prediction2']['score'] is num
                ? json['prediction2']['score'].toDouble()
                : 0.0),
        'category':
            json['prediction2']['category']?.toString() ?? 'Not Assigned',
        'recommendations':
            (json['prediction2']['recommendations'] as List<dynamic>?)
                ?.map(
                  (rec) => JewelryRecommendation.fromJson(
                    rec as Map<String, dynamic>,
                  ),
                )
                .toList() ??
            [],
        'overall_feedback':
            json['prediction2']['overall_feedback']?.toString() ??
            'Not Provided',
        'feedback_required':
            json['prediction2']['feedback_required']?.toString(),
        'face_image_path': json['face_image_path']?.toString(),
        'jewelry_image_path': json['jewelry_image_path']?.toString(),
        'model': 'prediction2',
      },
      faceImagePath: json['face_image_path']?.toString(),
      jewelryImagePath: json['jewelry_image_path']?.toString(),
      validationStatus: json['validation_status']?.toString() ?? 'unknown',
      predictionStatus: json['prediction_status']?.toString() ?? 'unknown',
    );
  }
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  _HistoryScreenState createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final List<HistoryItem> _history = [];
  bool _isLoading = true;
  String? _errorMessage;
  final int _limit = 10;
  int _skip = 0;
  bool _hasMore = true;

  @override
  void initState() {
    super.initState();
    _fetchHistory();
  }

  Future<void> _fetchHistory({bool loadMore = false}) async {
    if (loadMore) {
      setState(() {
        _skip += _limit;
      });
    } else {
      setState(() {
        _isLoading = true;
        _errorMessage = null;
        _skip = 0;
        _history.clear();
        _hasMore = true;
      });
    }

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

    try {
      final response = await http
          .get(
            Uri.parse(
              'https://jewelify-server.onrender.com/history/?limit=$_limit&skip=$_skip',
            ),
            headers: {'Authorization': 'Bearer $token'},
          )
          .timeout(
            const Duration(seconds: 100),
            onTimeout: () {
              throw Exception("Request timed out while fetching history");
            },
          );

      if (response.statusCode == 200) {
        final dynamic data = json.decode(response.body);
        if (data is List) {
          final newItems =
              data
                  .map(
                    (item) =>
                        HistoryItem.fromJson(item as Map<String, dynamic>),
                  )
                  .toList();
          setState(() {
            _history.addAll(newItems);
            _isLoading = false;
            _hasMore = newItems.length == _limit;
          });
        } else if (data is Map && data.containsKey('message')) {
          setState(() {
            _isLoading = false;
            _hasMore = false;
          });
        }
      } else if (response.statusCode == 401) {
        setState(() {
          _errorMessage = "Session expired. Please log in again.";
          _isLoading = false;
        });
        authProvider.logout();
        Navigator.pushReplacementNamed(context, '/login');
      } else {
        setState(() {
          _errorMessage = "Failed to fetch history: ${response.statusCode}";
          _isLoading = false;
          _hasMore = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage =
            e.toString().contains("timed out")
                ? "Request timed out. Please check your connection and try again."
                : "Error fetching history: $e";
        _isLoading = false;
        _hasMore = false;
      });
    }
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

      final response = await request.send().timeout(
        const Duration(seconds: 100),
        onTimeout: () {
          throw Exception("Request timed out while submitting feedback");
        },
      );
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode != 200) {
        print(
          'Failed to submit feedback: ${response.statusCode} - $responseBody',
        );
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Failed to submit feedback. Please try again."),
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Feedback submitted successfully!")),
        );
      }
    } catch (e) {
      print('Error submitting feedback: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Error submitting feedback. Please try again."),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFEDE7F6),
      appBar: AppBar(
        backgroundColor: const Color(0xFFEDE7F6),
        foregroundColor: Colors.black,
        title: const Text('History'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Stack(
        children: [
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Your Prediction History',
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Expanded(
                    child:
                        _isLoading && _history.isEmpty
                            ? const Center(child: CircularProgressIndicator())
                            : _errorMessage != null
                            ? Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Text(
                                    _errorMessage!,
                                    style: const TextStyle(color: Colors.red),
                                    textAlign: TextAlign.center,
                                  ),
                                  const SizedBox(height: 20),
                                  ElevatedButton(
                                    onPressed: () => _fetchHistory(),
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: const Color(0xFFEDE7F6),
                                      foregroundColor: Colors.black,
                                      shape: RoundedRectangleBorder(
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                    ),
                                    child: const Text('Retry'),
                                  ),
                                ],
                              ),
                            )
                            : _history.isEmpty
                            ? const Center(child: Text('No history exists'))
                            : RefreshIndicator(
                              onRefresh: () => _fetchHistory(),
                              child: ListView.builder(
                                cacheExtent: 1000,
                                itemCount: _history.length + (_hasMore ? 1 : 0),
                                itemBuilder: (context, index) {
                                  if (index == _history.length && _hasMore) {
                                    _fetchHistory(loadMore: true);
                                    return const Center(
                                      child: Padding(
                                        padding: EdgeInsets.all(16.0),
                                        child: CircularProgressIndicator(),
                                      ),
                                    );
                                  }

                                  final item = _history[index];
                                  return ExpandableItem(
                                    header: Padding(
                                      padding: const EdgeInsets.all(16.0),
                                      child: Row(
                                        children: [
                                          Expanded(
                                            child: Column(
                                              crossAxisAlignment:
                                                  CrossAxisAlignment.start,
                                              children: [
                                                Text(
                                                  item.date,
                                                  style: Theme.of(context)
                                                      .textTheme
                                                      .titleMedium
                                                      ?.copyWith(
                                                        fontWeight:
                                                            FontWeight.bold,
                                                      ),
                                                ),
                                                const SizedBox(height: 4),
                                                Text(
                                                  'Validation: ${item.validationStatus}',
                                                  style: Theme.of(
                                                    context,
                                                  ).textTheme.bodyLarge?.copyWith(
                                                    color:
                                                        item.validationStatus ==
                                                                'failed'
                                                            ? Colors.red
                                                            : null,
                                                  ),
                                                ),
                                                Text(
                                                  'Prediction: ${item.predictionStatus}',
                                                  style: Theme.of(
                                                    context,
                                                  ).textTheme.bodyLarge?.copyWith(
                                                    color:
                                                        item.predictionStatus ==
                                                                'failed'
                                                            ? Colors.red
                                                            : null,
                                                  ),
                                                ),
                                                const SizedBox(height: 4),
                                                Text(
                                                  'Model 1: ${item.prediction1['category']}',
                                                  style:
                                                      Theme.of(
                                                        context,
                                                      ).textTheme.bodyLarge,
                                                ),
                                                const SizedBox(height: 4),
                                                Text(
                                                  'Score: ${item.prediction1['score'].toStringAsFixed(1)}%',
                                                  style:
                                                      Theme.of(
                                                        context,
                                                      ).textTheme.bodyLarge,
                                                ),
                                                const SizedBox(height: 4),
                                                Text(
                                                  'Model 2: ${item.prediction2['category']}',
                                                  style:
                                                      Theme.of(
                                                        context,
                                                      ).textTheme.bodyLarge,
                                                ),
                                                const SizedBox(height: 4),
                                                Text(
                                                  'Score: ${item.prediction2['score'].toStringAsFixed(1)}%',
                                                  style:
                                                      Theme.of(
                                                        context,
                                                      ).textTheme.bodyLarge,
                                                ),
                                              ],
                                            ),
                                          ),
                                          Icon(
                                            item.isExpanded
                                                ? Icons.keyboard_arrow_up
                                                : Icons.keyboard_arrow_down,
                                            color: Colors.black,
                                          ),
                                        ],
                                      ),
                                    ),
                                    content: Column(
                                      children: [
                                        PredictionModule(
                                          modelName: "Model 1 (XGBoost)",
                                          prediction: item.prediction1,
                                          predictionId: item.id,
                                          onFeedbackSubmit: _submitFeedback,
                                        ),
                                        const SizedBox(height: 16),
                                        PredictionModule(
                                          modelName: "Model 2 (MLP)",
                                          prediction: item.prediction2,
                                          predictionId: item.id,
                                          onFeedbackSubmit: _submitFeedback,
                                        ),
                                      ],
                                    ),
                                    isExpanded: item.isExpanded,
                                    onToggle:
                                        (expanded) => setState(
                                          () => item.isExpanded = expanded,
                                        ),
                                  );
                                },
                              ),
                            ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
