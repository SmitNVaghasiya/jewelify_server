import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

class ImageStorage {
  static Future<String?> saveImage(File image, String fileName) async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final filePath = path.join(directory.path, '$fileName.jpg');
      await image.copy(filePath);
      return filePath;
    } catch (e) {
      print('Error saving image: $e');
      return null;
    }
  }

  static Future<File?> getImage(String? imagePath) async {
    if (imagePath == null) return null;
    final file = File(imagePath);
    if (await file.exists()) {
      return file;
    }
    return null;
  }

  // Cache to store resolved image files
  static final Map<String, File?> _imageCache = {};

  static Future<File?> getCachedImage(String? imagePath) async {
    if (imagePath == null) return null;
    if (_imageCache.containsKey(imagePath)) {
      return _imageCache[imagePath];
    }
    final file = await getImage(imagePath);
    _imageCache[imagePath] = file;
    return file;
  }
}
