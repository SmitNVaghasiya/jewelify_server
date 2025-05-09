// import 'package:flutter/material.dart';
// import 'package:cached_network_image/cached_network_image.dart';
// import '../models/jewelry_recommendation.dart';

// class RecommendationCard extends StatelessWidget {
//   final JewelryRecommendation recommendation;
//   final Function(String?) onImageTap;

//   const RecommendationCard({
//     super.key,
//     required this.recommendation,
//     required this.onImageTap,
//   });

//   String addEmojiToCategory(String category) {
//     switch (category.trim().toLowerCase()) {
//       case 'very good':
//         return '⭐ Very Good';
//       case 'good':
//         return '✅ Good';
//       case 'neutral':
//         return '😐 Neutral';
//       case 'bad':
//         return '⚠️ Bad';
//       case 'very bad':
//         return '❌ Very Bad';
//       default:
//         return category;
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     final theme = Theme.of(context);
//     return Padding(
//       padding: const EdgeInsets.symmetric(vertical: 8.0),
//       child: Column(
//         crossAxisAlignment: CrossAxisAlignment.start,
//         children: [
//           Text(
//             recommendation.name,
//             style: theme.textTheme.bodyLarge?.copyWith(
//               fontWeight: FontWeight.w600,
//             ),
//           ),
//           const SizedBox(height: 8),
//           GestureDetector(
//             onTap: () => onImageTap(recommendation.displayUrl),
//             child: ClipRRect(
//               borderRadius: BorderRadius.circular(12),
//               child: CachedNetworkImage(
//                 imageUrl: recommendation.displayUrl,
//                 width: double.infinity,
//                 height: 150,
//                 fit: BoxFit.cover,
//                 placeholder:
//                     (context, url) =>
//                         const Center(child: CircularProgressIndicator()),
//                 errorWidget:
//                     (context, url, error) => Container(
//                       width: double.infinity,
//                       height: 150,
//                       color: Colors.grey[300],
//                       child: const Icon(Icons.error),
//                     ),
//               ),
//             ),
//           ),
//           const SizedBox(height: 8),
//           Text(
//             "Compatibility Score: ${(recommendation.score).toStringAsFixed(1)}%",
//             style: theme.textTheme.bodyLarge,
//           ),
//           const SizedBox(height: 4),
//           Text(
//             "Category: ${addEmojiToCategory(recommendation.category)}",
//             style: theme.textTheme.bodyLarge?.copyWith(
//               fontFamily: 'NotoColorEmoji',
//             ),
//           ),
//         ],
//       ),
//     );
//   }
// }

import 'package:flutter/material.dart';
import '../models/jewelry_recommendation.dart';

class RecommendationCard extends StatelessWidget {
  final JewelryRecommendation recommendation;
  final void Function(String url)? onImageTap;

  const RecommendationCard({
    super.key,
    required this.recommendation,
    this.onImageTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Row(
          children: [
            if (recommendation.displayUrl != null &&
                recommendation.displayUrl!.isNotEmpty)
              GestureDetector(
                onTap: onImageTap != null
                    ? () => onImageTap!(recommendation.displayUrl!)
                    : null,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.network(
                    recommendation.displayUrl!,
                    width: 80,
                    height: 80,
                    fit: BoxFit.cover,
                    errorBuilder: (context, error, stackTrace) {
                      return Container(
                        width: 80,
                        height: 80,
                        color: Colors.grey[300],
                        child: const Icon(Icons.image_not_supported),
                      );
                    },
                  ),
                ),
              )
            else
              Container(
                width: 80,
                height: 80,
                color: Colors.grey[300],
                child: const Icon(Icons.image_not_supported),
              ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    recommendation.name,
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    "Compatibility Score: ${recommendation.score.toStringAsFixed(2)}%",
                    style: theme.textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    "Category: ${recommendation.category}",
                    style: theme.textTheme.bodyMedium,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}