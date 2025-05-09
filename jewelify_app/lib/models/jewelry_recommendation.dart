class JewelryRecommendation {
  final String name;
  final String? displayUrl;
  final double score;
  final String category;
  final bool? liked;
  final double? userScore;

  JewelryRecommendation({
    required this.name,
    this.displayUrl,
    required this.score,
    required this.category,
    this.liked,
    this.userScore,
  });

  factory JewelryRecommendation.fromJson(Map<String, dynamic> json) {
    return JewelryRecommendation(
      name: json['name'] as String,
      displayUrl: json['display_url'] as String?,
      score: (json['score'] is num ? json['score'].toDouble() : 0.0),
      category: json['category'] as String,
      liked: json['liked'] as bool?,
      userScore:
          json['user_score'] != null
              ? (json['user_score'] is num
                  ? json['user_score'].toDouble()
                  : null)
              : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'display_url': displayUrl,
      'score': score,
      'category': category,
      'liked': liked,
      'user_score': userScore,
    };
  }
}
