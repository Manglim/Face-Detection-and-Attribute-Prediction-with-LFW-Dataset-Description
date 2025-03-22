# Face Gender and Hair Color Predictor

## Overview

`face_gender_hair_predictor.py` is a Python script that processes images to:
- Detect the presence of a face using OpenCV's Haar Cascade classifier.
- Predict the gender (man or woman) of the detected face using the `deepface` library.
- Predict the hair color (e.g., black, brown, blonde, red, gray) using MediaPipe for hair segmentation and K-Means clustering for color analysis.

The script is designed to work with the Labeled Faces in the Wild (LFW) dataset but can be adapted for other image datasets. It processes a sample of images, displays the results with bounding boxes around detected faces, and prints a summary of the predictions.

## Features

- **Face Detection**: Uses OpenCV's Haar Cascade classifier to determine if a face is present in an image.
- **Gender Prediction**: Uses the `deepface` library to predict the gender of the detected face (outputs "Man" or "Woman").
- **Hair Color Prediction**: 
  - Segments the hair region using MediaPipe's Selfie Segmentation model.
  - Applies K-Means clustering to identify the dominant color in the hair region.
  - Maps the dominant color to common hair colors (black, brown, blonde, red, gray) using HSV color space.
- **Visualization**: Displays sample images with bounding boxes around detected faces, along with the predicted face presence, gender, and hair color.
- **Summary**: Prints a summary of the predictions for each processed image.

## Dependencies

To run the script, you need the following Python libraries:

- `opencv-python` (for face detection and image processing)
- `deepface` (for gender prediction)
- `mediapipe` (for hair segmentation)
- `scikit-learn` (for K-Means clustering)
- `matplotlib` (for visualization)
- `numpy` (for numerical operations)

You can install the dependencies using pip:

```bash
pip install opencv-python deepface mediapipe scikit-learn matplotlib numpy
