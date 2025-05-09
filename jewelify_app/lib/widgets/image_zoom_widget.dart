// import 'dart:io';
// import 'package:flutter/material.dart';
// import 'package:photo_view/photo_view.dart';
// import 'package:cached_network_image/cached_network_image.dart';
// import 'package:smooth_page_indicator/smooth_page_indicator.dart';

// class ZoomableImage extends StatefulWidget {
//   final int initialIndex;
//   final List<Map<String, dynamic>> images;
//   final Function() onClose;

//   const ZoomableImage({
//     super.key,
//     required this.initialIndex,
//     required this.images,
//     required this.onClose,
//   });

//   @override
//   _ZoomableImageState createState() => _ZoomableImageState();
// }

// class _ZoomableImageState extends State<ZoomableImage> {
//   late PageController _pageController;
//   late List<PhotoViewController> _photoViewControllers;

//   @override
//   void initState() {
//     super.initState();
//     _pageController = PageController(initialPage: widget.initialIndex);
//     _photoViewControllers = List.generate(
//       widget.images.length,
//       (index) => PhotoViewController(),
//     );
//   }

//   @override
//   void dispose() {
//     _pageController.dispose();
//     for (var controller in _photoViewControllers) {
//       controller.dispose();
//     }
//     super.dispose();
//   }

//   @override
//   Widget build(BuildContext context) {
//     return GestureDetector(
//       onTap: widget.onClose,
//       child: Container(
//         color: Colors.black54,
//         child: Stack(
//           children: [
//             Center(
//               child: SizedBox(
//                 width: double.infinity,
//                 height: double.infinity,
//                 child: PageView.builder(
//                   controller: _pageController,
//                   itemCount: widget.images.length,
//                   itemBuilder: (context, index) {
//                     final image = widget.images[index];
//                     if (image['localFileFuture'] != null) {
//                       return FutureBuilder<File?>(
//                         future: image['localFileFuture'] as Future<File?>,
//                         builder: (context, snapshot) {
//                           if (snapshot.connectionState ==
//                               ConnectionState.waiting) {
//                             return const Center(
//                               child: CircularProgressIndicator(),
//                             );
//                           }
//                           if (snapshot.hasData && snapshot.data != null) {
//                             return PhotoView(
//                               controller: _photoViewControllers[index],
//                               imageProvider: FileImage(snapshot.data!),
//                               minScale: PhotoViewComputedScale.contained,
//                               maxScale: PhotoViewComputedScale.covered * 2,
//                               initialScale: PhotoViewComputedScale.contained,
//                             );
//                           }
//                           return const Center(
//                             child: Icon(
//                               Icons.error,
//                               color: Colors.white,
//                               size: 40,
//                             ),
//                           );
//                         },
//                       );
//                     } else if (image['imageUrl'] != null) {
//                       return PhotoView(
//                         controller: _photoViewControllers[index],
//                         imageProvider: CachedNetworkImageProvider(
//                           image['imageUrl'] as String,
//                         ),
//                         minScale: PhotoViewComputedScale.contained,
//                         maxScale: PhotoViewComputedScale.covered * 2,
//                         initialScale: PhotoViewComputedScale.contained,
//                         errorBuilder:
//                             (context, error, stackTrace) => const Center(
//                               child: Icon(
//                                 Icons.error,
//                                 color: Colors.white,
//                                 size: 40,
//                               ),
//                             ),
//                       );
//                     }
//                     return const Center(
//                       child: Icon(Icons.error, color: Colors.white, size: 40),
//                     );
//                   },
//                 ),
//               ),
//             ),
//             Positioned(
//               top: 40,
//               right: 10,
//               child: IconButton(
//                 icon: const Icon(Icons.close, color: Colors.white, size: 40),
//                 onPressed: widget.onClose,
//               ),
//             ),
//             Positioned(
//               bottom: 20,
//               left: 0,
//               right: 0,
//               child: Center(
//                 child: SmoothPageIndicator(
//                   controller: _pageController,
//                   count: widget.images.length,
//                   effect: const WormEffect(
//                     dotHeight: 8,
//                     dotWidth: 8,
//                     activeDotColor: Colors.white,
//                     dotColor: Colors.white38,
//                   ),
//                 ),
//               ),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }

import 'dart:io';
import 'package:flutter/material.dart';

class ZoomableImage extends StatefulWidget {
  final int initialIndex;
  final List<Map<String, dynamic>> images;
  final VoidCallback onClose;

  const ZoomableImage({
    super.key,
    required this.initialIndex,
    required this.images,
    required this.onClose,
  });

  @override
  _ZoomableImageState createState() => _ZoomableImageState();
}

class _ZoomableImageState extends State<ZoomableImage> {
  late PageController _pageController;
  late int _currentIndex;

  @override
  void initState() {
    super.initState();
    _currentIndex = widget.initialIndex;
    _pageController = PageController(initialPage: _currentIndex);
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.black,
      insetPadding: EdgeInsets.zero,
      child: Stack(
        children: [
          PageView.builder(
            controller: _pageController,
            itemCount: widget.images.length,
            onPageChanged: (index) {
              setState(() {
                _currentIndex = index;
              });
            },
            itemBuilder: (context, index) {
              final imageData = widget.images[index];
              if (imageData['localFileFuture'] != null) {
                return FutureBuilder<File?>(
                  future: imageData['localFileFuture'] as Future<File?>,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const Center(child: CircularProgressIndicator());
                    }
                    if (snapshot.hasData && snapshot.data != null) {
                      return InteractiveViewer(
                        child: Image.file(snapshot.data!, fit: BoxFit.contain),
                      );
                    }
                    return const Center(
                      child: Icon(
                        Icons.image_not_supported,
                        color: Colors.white,
                      ),
                    );
                  },
                );
              } else if (imageData['networkUrl'] != null) {
                return InteractiveViewer(
                  child: Image.network(
                    imageData['networkUrl'] as String,
                    fit: BoxFit.contain,
                    errorBuilder: (context, error, stackTrace) {
                      return const Center(
                        child: Icon(
                          Icons.image_not_supported,
                          color: Colors.white,
                        ),
                      );
                    },
                  ),
                );
              }
              return const Center(
                child: Icon(Icons.image_not_supported, color: Colors.white),
              );
            },
          ),
          Positioned(
            top: 40,
            right: 16,
            child: IconButton(
              icon: const Icon(Icons.close, color: Colors.white),
              onPressed: widget.onClose,
            ),
          ),
          if (widget.images.length > 1)
            Positioned(
              bottom: 16,
              left: 0,
              right: 0,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: List.generate(
                  widget.images.length,
                  (index) => Container(
                    margin: const EdgeInsets.symmetric(horizontal: 4),
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color:
                          _currentIndex == index ? Colors.white : Colors.grey,
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
