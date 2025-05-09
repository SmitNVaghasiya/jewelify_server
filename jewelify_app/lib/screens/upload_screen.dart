import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:developer' as developer;

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  _UploadScreenState createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  File? _faceImage;
  File? _jewelryImage;
  final ImagePicker _picker = ImagePicker();
  bool _isLoading = false;

  Future<void> _pickImage(String type) async {
    try {
      final source = await _showImageSourceBottomSheet();
      if (source == null) return;

      final pickedFile = await _picker.pickImage(source: source);

      if (pickedFile != null) {
        final file = File(pickedFile.path);
        if (await file.exists()) {
          // Validate image size (limit to 5MB)
          final fileSize = await file.length();
          const maxSizeInBytes = 5 * 1024 * 1024; // 5MB
          if (fileSize > maxSizeInBytes) {
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text(
                    "Image size too large. Please select an image under 5MB.",
                  ),
                ),
              );
            }
            return;
          }

          // Validate image format
          final extension = pickedFile.path.split('.').last.toLowerCase();
          if (!['jpg', 'jpeg', 'png'].contains(extension)) {
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text(
                    "Unsupported image format. Please use JPG or PNG.",
                  ),
                ),
              );
            }
            return;
          }

          if (mounted) {
            setState(() {
              if (type == 'face') {
                _faceImage = file;
              } else {
                _jewelryImage = file;
              }
            });
          }
        } else {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text("Failed to load image.")),
            );
          }
        }
      }
    } catch (e) {
      developer.log('Error picking image for $type: $e', name: 'UploadScreen');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Error picking image: $e")));
      }
    }
  }

  Future<ImageSource?> _showImageSourceBottomSheet() async {
    return showModalBottomSheet<ImageSource>(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder:
          (context) => Container(
            padding: const EdgeInsets.symmetric(vertical: 25, horizontal: 16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const SizedBox(height: 15),
                ListTile(
                  leading: const Icon(Icons.photo_library, size: 28),
                  title: const Text("Gallery"),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 24),
                  onTap: () => Navigator.pop(context, ImageSource.gallery),
                ),
                const Divider(),
                const SizedBox(height: 15),
                ListTile(
                  leading: const Icon(Icons.camera_alt, size: 28),
                  title: const Text("Camera"),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 24),
                  onTap: () => Navigator.pop(context, ImageSource.camera),
                ),
                const SizedBox(height: 15),
              ],
            ),
          ),
    );
  }

  Future<bool> _onWillPop() async {
    if (_isLoading) {
      return await showDialog(
            context: context,
            builder:
                (context) => AlertDialog(
                  title: const Text("Cancel Upload?"),
                  content: const Text(
                    "Are you sure you want to cancel the upload process?",
                  ),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context, false),
                      child: const Text("No"),
                    ),
                    TextButton(
                      onPressed: () => Navigator.pop(context, true),
                      child: const Text("Yes"),
                    ),
                  ],
                ),
          ) ??
          false;
    }
    return true;
  }

  Future<void> _uploadImages() async {
    if (_isLoading) return;

    if (_faceImage == null || _jewelryImage == null) {
      developer.log(
        'Missing images: faceImage=$_faceImage, jewelryImage=$_jewelryImage',
        name: 'UploadScreen',
      );
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please upload both images')),
      );
      return;
    }

    developer.log(
      'Starting upload with faceImage=$_faceImage, jewelryImage=$_jewelryImage',
      name: 'UploadScreen',
    );
    setState(() => _isLoading = true);

    try {
      if (mounted) {
        developer.log('Navigating to /processing', name: 'UploadScreen');
        await Navigator.pushNamed(
          context,
          '/processing',
          arguments: {'face': _faceImage, 'jewelry': _jewelryImage},
        );
        if (mounted) {
          developer.log('Returned from processing', name: 'UploadScreen');
          setState(() {
            _isLoading = false;
            _faceImage = null;
            _jewelryImage = null;
          });
        }
      }
    } catch (e) {
      developer.log('Error during navigation: $e', name: 'UploadScreen');
      if (mounted) {
        setState(() => _isLoading = false);
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Error during processing: $e")));
      }
    }
  }

  Widget _buildImageCard({
    required String title,
    required String subtitle,
    required IconData icon,
    required File? image,
    required VoidCallback onTap,
  }) {
    final theme = Theme.of(context);
    return Card(
      elevation: 4,
      margin: const EdgeInsets.symmetric(vertical: 12),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(20),
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            children: [
              image == null
                  ? Container(
                    width: double.infinity,
                    height: 180,
                    decoration: BoxDecoration(
                      color: theme.colorScheme.primary.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Icon(
                      icon,
                      size: 70,
                      color: theme.colorScheme.primary.withOpacity(0.7),
                    ),
                  )
                  : ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: Image.file(
                      image,
                      width: double.infinity,
                      height: 180,
                      fit: BoxFit.cover,
                    ),
                  ),
              const SizedBox(height: 16),
              Text(
                title,
                style: theme.textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                subtitle,
                style: theme.textTheme.bodyMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: onTap,
                icon: Icon(
                  image == null ? Icons.add_photo_alternate : Icons.edit,
                ),
                label: Text(image == null ? "Select Image" : "Change Image"),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 12,
                  ),
                  side: BorderSide(color: theme.colorScheme.primary),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final bool canSubmit = _faceImage != null && _jewelryImage != null;

    return WillPopScope(
      onWillPop: _onWillPop,
      child: Scaffold(
        appBar: AppBar(
          title: const Text(
            "Upload Images",
            style: TextStyle(fontWeight: FontWeight.w600),
          ),
          elevation: 0,
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
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(
                    child: SingleChildScrollView(
                      child: Column(
                        children: [
                          _buildImageCard(
                            title: "Your Face Photo",
                            subtitle: "Upload a clear image of your face",
                            icon: Icons.face,
                            image: _faceImage,
                            onTap: () => _pickImage('face'),
                          ),
                          _buildImageCard(
                            title: "Jewelry Photo",
                            subtitle: "Upload the jewelry you want to try",
                            icon: Icons.diamond,
                            image: _jewelryImage,
                            onTap: () => _pickImage('jewelry'),
                          ),
                        ],
                      ),
                    ),
                  ),
                  ElevatedButton(
                    onPressed: canSubmit && !_isLoading ? _uploadImages : null,
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 18),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                      elevation: 6,
                    ),
                    child:
                        _isLoading
                            ? const CircularProgressIndicator(
                              color: Colors.white,
                            )
                            : const Text(
                              "Match Jewelry",
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
