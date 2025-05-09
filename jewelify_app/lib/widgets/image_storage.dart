import 'dart:io';
import 'package:path_provider/path_provider.dart';

class ImageStorage {
  static Future<File?> getCachedImage(String? imagePath) async {
    if (imagePath == null || imagePath.isEmpty) return null;

    try {
      final directory = await getTemporaryDirectory();
      final fileName = imagePath.split('/').last;
      final file = File('${directory.path}/$fileName');
      if (await file.exists()) {
        return file;
      }
      return null;
    } catch (e) {
      print('Error getting cached image: $e');
      return null;
    }
  }
}
