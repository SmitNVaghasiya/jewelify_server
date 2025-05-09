// import 'dart:io';
// import 'package:flutter/material.dart';
// import '../screens/image_storage.dart';
// import '../models/jewelry_recommendation.dart';
// import 'expandable_item.dart';
// import 'image_zoom_widget.dart';
// import 'recommendation_card.dart';
// import 'score_display.dart';

// class PredictionModule extends StatefulWidget {
//   final String modelName;
//   final Map<String, dynamic> prediction;
//   final String predictionId;
//   final Future<void> Function(
//     String,
//     String, {
//     String? recommendationName,
//     String review,
//   })
//   onFeedbackSubmit;

//   const PredictionModule({
//     super.key,
//     required this.modelName,
//     required this.prediction,
//     required this.predictionId,
//     required this.onFeedbackSubmit,
//   });

//   @override
//   _PredictionModuleState createState() => _PredictionModuleState();
// }

// class _PredictionModuleState extends State<PredictionModule>
//     with AutomaticKeepAliveClientMixin {
//   bool _isExpanded = false;
//   late Future<File?> _faceImageFuture;
//   late Future<File?> _jewelryImageFuture;
//   double _overallFeedbackScore = 0.0;
//   final Map<String, double> _recommendationFeedbackScores = {};
//   bool _hasSubmittedOverallFeedback = false;
//   bool _isSubmittingFeedback = false;

//   @override
//   void initState() {
//     super.initState();
//     _faceImageFuture = ImageStorage.getCachedImage(
//       widget.prediction['face_image_path'],
//     );
//     _jewelryImageFuture = ImageStorage.getCachedImage(
//       widget.prediction['jewelry_image_path'],
//     );

//     // Correctly cast recommendations to List<JewelryRecommendation>
//     final recommendations =
//         widget.prediction['recommendations'] as List<JewelryRecommendation>? ??
//         [];
//     for (var rec in recommendations) {
//       _recommendationFeedbackScores[rec.name] = 0.0;
//     }

//     if (widget.prediction['overall_feedback'] != null &&
//         widget.prediction['overall_feedback'] != 'Not Provided') {
//       _overallFeedbackScore =
//           double.tryParse(widget.prediction['overall_feedback'].toString()) ??
//           0.0;
//       _hasSubmittedOverallFeedback = true;
//     }

//     // Check if feedback is required
//     if (widget.prediction['feedback_required'] == false) {
//       _hasSubmittedOverallFeedback = true;
//     }
//   }

//   void _showZoomableImage(int initialIndex, List<Map<String, dynamic>> images) {
//     showDialog(
//       context: context,
//       builder:
//           (context) => ZoomableImage(
//             initialIndex: initialIndex,
//             images: images,
//             onClose: () => Navigator.of(context).pop(),
//           ),
//     );
//   }

//   void _submitOverallFeedback(double score) async {
//     if (_hasSubmittedOverallFeedback || _isSubmittingFeedback) return;

//     setState(() {
//       _isSubmittingFeedback = true;
//     });

//     String review = (score / 100.0).toStringAsFixed(2);
//     await widget.onFeedbackSubmit(
//       widget.predictionId,
//       widget.prediction['model'],
//       review: review,
//     );
//     if (mounted) {
//       setState(() {
//         _hasSubmittedOverallFeedback = true;
//         _isSubmittingFeedback = false;
//       });
//     }
//   }

//   void _submitRecommendationFeedback(
//     String recommendationName,
//     double score,
//   ) async {
//     if (_isSubmittingFeedback) return;

//     setState(() {
//       _isSubmittingFeedback = true;
//     });

//     String review = (score / 100.0).toStringAsFixed(2);
//     await widget.onFeedbackSubmit(
//       widget.predictionId,
//       widget.prediction['model'],
//       recommendationName: recommendationName,
//       review: review,
//     );
//     if (mounted) {
//       setState(() {
//         _isSubmittingFeedback = false;
//       });
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     super.build(context); // Required for AutomaticKeepAliveClientMixin
//     final theme = Theme.of(context);
//     final score = widget.prediction['score'] as num? ?? 0.0;
//     final category =
//         widget.prediction['category']?.toString() ?? 'Not Assigned';
//     final recommendations =
//         widget.prediction['recommendations'] as List<JewelryRecommendation>? ??
//         [];

//     // Collect all images for the gallery
//     final List<Map<String, dynamic>> images = [];
//     if (widget.prediction['face_image_path'] != null) {
//       images.add({'localFileFuture': _faceImageFuture});
//     }
//     if (widget.prediction['jewelry_image_path'] != null) {
//       images.add({'localFileFuture': _jewelryImageFuture});
//     }
//     for (var rec in recommendations) {
//       images.add({'imageUrl': rec.displayUrl});
//     }

//     return ExpandableItem(
//       header: Padding(
//         padding: const EdgeInsets.all(16.0),
//         child: Row(
//           children: [
//             Expanded(
//               child: Text(widget.modelName, style: theme.textTheme.titleLarge),
//             ),
//             Icon(
//               _isExpanded ? Icons.keyboard_arrow_up : Icons.keyboard_arrow_down,
//               color: theme.colorScheme.primary,
//             ),
//           ],
//         ),
//       ),
//       content: Padding(
//         padding: const EdgeInsets.all(16.0),
//         child: Column(
//           crossAxisAlignment: CrossAxisAlignment.start,
//           children: [
//             if (widget.prediction['face_image_path'] != null ||
//                 widget.prediction['jewelry_image_path'] != null)
//               Column(
//                 crossAxisAlignment: CrossAxisAlignment.start,
//                 children: [
//                   Text(
//                     "Uploaded Images",
//                     style: theme.textTheme.titleLarge?.copyWith(
//                       fontWeight: FontWeight.bold,
//                     ),
//                   ),
//                   const SizedBox(height: 12),
//                   Row(
//                     mainAxisAlignment: MainAxisAlignment.spaceEvenly,
//                     children: [
//                       if (widget.prediction['face_image_path'] != null)
//                         FutureBuilder<File?>(
//                           future: _faceImageFuture,
//                           builder: (context, snapshot) {
//                             if (snapshot.connectionState ==
//                                 ConnectionState.waiting) {
//                               return const CircularProgressIndicator();
//                             }
//                             if (snapshot.hasData && snapshot.data != null) {
//                               return GestureDetector(
//                                 onTap: () => _showZoomableImage(0, images),
//                                 child: ClipRRect(
//                                   borderRadius: BorderRadius.circular(12),
//                                   child: Image.file(
//                                     snapshot.data!,
//                                     width: 100,
//                                     height: 100,
//                                     fit: BoxFit.cover,
//                                   ),
//                                 ),
//                               );
//                             }
//                             return Container(
//                               width: 100,
//                               height: 100,
//                               color: Colors.grey[300],
//                               child: const Icon(Icons.image_not_supported),
//                             );
//                           },
//                         ),
//                       if (widget.prediction['jewelry_image_path'] != null)
//                         FutureBuilder<File?>(
//                           future: _jewelryImageFuture,
//                           builder: (context, snapshot) {
//                             if (snapshot.connectionState ==
//                                 ConnectionState.waiting) {
//                               return const CircularProgressIndicator();
//                             }
//                             if (snapshot.hasData && snapshot.data != null) {
//                               return GestureDetector(
//                                 onTap: () => _showZoomableImage(1, images),
//                                 child: ClipRRect(
//                                   borderRadius: BorderRadius.circular(12),
//                                   child: Image.file(
//                                     snapshot.data!,
//                                     width: 100,
//                                     height: 100,
//                                     fit: BoxFit.cover,
//                                   ),
//                                 ),
//                               );
//                             }
//                             return Container(
//                               width: 100,
//                               height: 100,
//                               color: Colors.grey[300],
//                               child: const Icon(Icons.image_not_supported),
//                             );
//                           },
//                         ),
//                     ],
//                   ),
//                   const SizedBox(height: 20),
//                 ],
//               ),
//             ScoreDisplay(score: score.toDouble(), category: category),
//             const SizedBox(height: 20),
//             Row(
//               mainAxisAlignment: MainAxisAlignment.spaceBetween,
//               children: [
//                 Text(
//                   "Overall Score",
//                   style: theme.textTheme.titleMedium?.copyWith(
//                     fontWeight: FontWeight.bold,
//                   ),
//                 ),
//                 Text(
//                   "${_overallFeedbackScore.round()}",
//                   style: theme.textTheme.titleMedium?.copyWith(
//                     fontWeight: FontWeight.bold,
//                   ),
//                 ),
//               ],
//             ),
//             Slider(
//               value: _overallFeedbackScore,
//               min: 0.0,
//               max: 100.0,
//               divisions: 100,
//               label: _overallFeedbackScore.round().toString(),
//               onChanged:
//                   _hasSubmittedOverallFeedback || _isSubmittingFeedback
//                       ? null
//                       : (value) {
//                         setState(() {
//                           _overallFeedbackScore = value;
//                         });
//                         _submitOverallFeedback(value);
//                       },
//             ),
//             const SizedBox(height: 20),
//             Text(
//               "Recommendations",
//               style: theme.textTheme.titleLarge?.copyWith(
//                 fontWeight: FontWeight.bold,
//               ),
//             ),
//             const SizedBox(height: 12),
//             if (recommendations.isNotEmpty)
//               ...recommendations.map(
//                 (rec) => Column(
//                   crossAxisAlignment: CrossAxisAlignment.start,
//                   children: [
//                     RecommendationCard(
//                       recommendation: rec,
//                       onImageTap: (url) {
//                         final recIndex = images.indexWhere(
//                           (img) => img['imageUrl'] == url,
//                         );
//                         if (recIndex != -1) {
//                           _showZoomableImage(recIndex, images);
//                         }
//                       },
//                     ),
//                     const SizedBox(height: 10),
//                     Row(
//                       mainAxisAlignment: MainAxisAlignment.spaceBetween,
//                       children: [
//                         Text(
//                           "Feedback Score",
//                           style: theme.textTheme.titleMedium?.copyWith(
//                             fontWeight: FontWeight.bold,
//                           ),
//                         ),
//                         Text(
//                           "${(_recommendationFeedbackScores[rec.name] ?? 0.0).round()}",
//                           style: theme.textTheme.titleMedium?.copyWith(
//                             fontWeight: FontWeight.bold,
//                           ),
//                         ),
//                       ],
//                     ),
//                     Slider(
//                       value: _recommendationFeedbackScores[rec.name] ?? 0.0,
//                       min: 0.0,
//                       max: 100.0,
//                       divisions: 100,
//                       label:
//                           (_recommendationFeedbackScores[rec.name] ?? 0.0)
//                               .round()
//                               .toString(),
//                       onChanged:
//                           _isSubmittingFeedback
//                               ? null
//                               : (value) {
//                                 setState(() {
//                                   _recommendationFeedbackScores[rec.name] =
//                                       value;
//                                 });
//                                 _submitRecommendationFeedback(rec.name, value);
//                               },
//                     ),
//                     const SizedBox(height: 10),
//                   ],
//                 ),
//               )
//             else
//               Text(
//                 "No recommendations available",
//                 style: theme.textTheme.bodyLarge?.copyWith(color: Colors.grey),
//               ),
//           ],
//         ),
//       ),
//       isExpanded: _isExpanded,
//       onToggle: (expanded) => setState(() => _isExpanded = expanded),
//     );
//   }

//   @override
//   bool get wantKeepAlive => true;
// }

import 'dart:io';
import 'package:flutter/material.dart';
import '../models/jewelry_recommendation.dart';
import 'image_storage.dart';
import 'image_zoom_widget.dart';
import 'recommendation_card.dart';
import 'score_display.dart';

class PredictionModule extends StatefulWidget {
  final String modelName;
  final Map<String, dynamic> prediction;
  final String predictionId;
  final Future<void> Function(
    String,
    String, {
    String? recommendationName,
    String review,
  })
  onFeedbackSubmit;

  const PredictionModule({
    super.key,
    required this.modelName,
    required this.prediction,
    required this.predictionId,
    required this.onFeedbackSubmit,
  });

  @override
  _PredictionModuleState createState() => _PredictionModuleState();
}

class _PredictionModuleState extends State<PredictionModule>
    with AutomaticKeepAliveClientMixin {
  late Future<File?> _faceImageFuture;
  late Future<File?> _jewelryImageFuture;
  double _overallFeedbackScore = 0.0;
  final Map<String, double> _recommendationFeedbackScores = {};
  bool _hasSubmittedOverallFeedback = false;
  bool _isSubmittingFeedback = false;

  @override
  void initState() {
    super.initState();
    _faceImageFuture = ImageStorage.getCachedImage(
      widget.prediction['face_image_path'],
    );
    _jewelryImageFuture = ImageStorage.getCachedImage(
      widget.prediction['jewelry_image_path'],
    );

    final recommendations =
        widget.prediction['recommendations'] as List<JewelryRecommendation>? ??
        [];
    for (var rec in recommendations) {
      _recommendationFeedbackScores[rec.name] = 0.0;
    }

    if (widget.prediction['overall_feedback'] != null &&
        widget.prediction['overall_feedback'] != 'Not Provided') {
      _overallFeedbackScore =
          double.tryParse(widget.prediction['overall_feedback'].toString()) ??
          0.0;
      _hasSubmittedOverallFeedback = true;
    }

    if (widget.prediction['feedback_required'] == false) {
      _hasSubmittedOverallFeedback = true;
    }
  }

  void _showZoomableImage(int initialIndex, List<Map<String, dynamic>> images) {
    showDialog(
      context: context,
      builder:
          (context) => ZoomableImage(
            initialIndex: initialIndex,
            images: images,
            onClose: () => Navigator.of(context).pop(),
          ),
    );
  }

  void _submitOverallFeedback(double score) async {
    if (_hasSubmittedOverallFeedback || _isSubmittingFeedback) return;

    setState(() {
      _isSubmittingFeedback = true;
    });

    String review = (score / 100.0).toStringAsFixed(2);
    await widget.onFeedbackSubmit(
      widget.predictionId,
      widget.prediction['model'],
      review: review,
    );
    if (mounted) {
      setState(() {
        _hasSubmittedOverallFeedback = true;
        _overallFeedbackScore = score;
        _isSubmittingFeedback = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final theme = Theme.of(context);
    final score = widget.prediction['score'] as num? ?? 0.0;
    final category =
        widget.prediction['category']?.toString() ?? 'Not Assigned';
    final recommendations =
        widget.prediction['recommendations'] as List<JewelryRecommendation>? ??
        [];

    // Collect all images for the gallery
    final List<Map<String, dynamic>> images = [];
    if (widget.prediction['face_image_path'] != null) {
      images.add({'localFileFuture': _faceImageFuture});
    }
    if (widget.prediction['jewelry_image_path'] != null) {
      images.add({'localFileFuture': _jewelryImageFuture});
    }

    return Card(
      elevation: 4,
      margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Model Name
            Text(
              "Model | (${widget.modelName})",
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),

            // Uploaded Images
            Text(
              "Uploaded Images",
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                FutureBuilder<File?>(
                  future: _faceImageFuture,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const CircularProgressIndicator();
                    }
                    if (snapshot.hasData && snapshot.data != null) {
                      return GestureDetector(
                        onTap: () => _showZoomableImage(0, images),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.file(
                            snapshot.data!,
                            width: 100,
                            height: 100,
                            fit: BoxFit.cover,
                          ),
                        ),
                      );
                    }
                    return Container(
                      width: 100,
                      height: 100,
                      color: Colors.grey[300],
                      child: const Icon(Icons.image_not_supported),
                    );
                  },
                ),
                FutureBuilder<File?>(
                  future: _jewelryImageFuture,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const CircularProgressIndicator();
                    }
                    if (snapshot.hasData && snapshot.data != null) {
                      return GestureDetector(
                        onTap:
                            () => _showZoomableImage(
                              widget.prediction['face_image_path'] != null
                                  ? 1
                                  : 0,
                              images,
                            ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.file(
                            snapshot.data!,
                            width: 100,
                            height: 100,
                            fit: BoxFit.cover,
                          ),
                        ),
                      );
                    }
                    return Container(
                      width: 100,
                      height: 100,
                      color: Colors.grey[300],
                      child: const Icon(Icons.image_not_supported),
                    );
                  },
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Score Display
            ScoreDisplay(score: score.toDouble(), category: category),
            const SizedBox(height: 16),

            // Overall Feedback
            Text(
              "Overall Score",
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            if (!_hasSubmittedOverallFeedback)
              Slider(
                value: _overallFeedbackScore,
                min: 0,
                max: 100,
                divisions: 100,
                label: _overallFeedbackScore.round().toString(),
                onChanged: (value) {
                  setState(() {
                    _overallFeedbackScore = value;
                  });
                },
              )
            else
              Text(
                "Feedback: ${_overallFeedbackScore.toStringAsFixed(2)}",
                style: theme.textTheme.bodyLarge,
              ),
            const SizedBox(height: 16),

            // Recommendations
            Text(
              "Recommendations",
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            if (recommendations.isEmpty)
              const Text("No recommendations available.")
            else
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: recommendations.length,
                itemBuilder: (context, index) {
                  final recommendation = recommendations[index];
                  return RecommendationCard(
                    recommendation: recommendation,
                    onImageTap: (url) {
                      showDialog(
                        context: context,
                        builder:
                            (context) => ZoomableImage(
                              initialIndex: 0,
                              images: [
                                {'networkUrl': url},
                              ],
                              onClose: () => Navigator.of(context).pop(),
                            ),
                      );
                    },
                  );
                },
              ),
          ],
        ),
      ),
    );
  }

  @override
  bool get wantKeepAlive => true;
}
