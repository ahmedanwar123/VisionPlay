# VisionPlay

This project aims to use **YOLO** and several computer vision techniques to detect objects, segment players, measure their speed, and analyze the scene's perspective. The following methods are employed:

1. **YOLO Object Detection**: Used to detect players in images and videos.
2. **K-Means Clustering**: Applied to segment players from the background and identify their t-shirt color.
3. **Optical Flow**: Used to measure camera movement and track object motion.
4. **Perspective Transformation**: Utilizes OpenCV's perspective transform to represent the scene's depth and perspective.
5. **Speed and Distance Measurement**: Calculates the player's speed and distance covered based on the detected movement.

## Requirements

To run this project, make sure you have the following libraries installed:

- Python 3.x
- YOLOv (Ultralytics)
- OpenCV
- NumPy
- Scikit-learn

## Features

- **Object Detection**: Uses YOLOv11 to detect players and other objects in real-time from image or video input.
- **Player Segmentation**: Segments players from the background using K-Means clustering and identifies t-shirt color.
- **Camera Movement Tracking**: Measures camera movement through optical flow analysis.
- **Scene Perspective**: Transforms the scene's perspective to represent depth using OpenCV.
- **Speed & Distance Calculation**: Measures the player's speed and distance covered on the field based on detected motion.
## Contributing

If you would like to contribute to this project, please fork the repository, make changes, and submit a pull request. Ensure that you follow the coding standards and provide necessary documentation for your changes.