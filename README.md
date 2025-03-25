# Face Detection and Attribute Prediction with LFW Dataset

## Overview
This project processes the Labeled Faces in the Wild (LFW) dataset to detect faces and predict gender and hair color. It extracts images from `lfw-funneled.tgz`, uses OpenCV for face detection, DeepFace for gender prediction, and MediaPipe with K-Means clustering for hair color prediction. Results are visualized with bounding boxes and attribute labels.

## Functionality
1. **Dataset Extraction**:
   - Extracts `lfw-funneled.tgz` into a working directory.

2. **Face Detection**:
   - Uses OpenCV's Haar Cascade to locate faces.

3. **Attribute Prediction**:
   - **Gender**: Predicted with DeepFace.
   - **Hair Color**: Segmented with MediaPipe Selfie Segmentation, analyzed via K-Means in HSV space.

4. **Visualization**:
   - Displays images with detected faces and predicted attributes.

## Frameworks and Libraries
- **Python**: Core.
- **NumPy**: Numerical ops (`np`).
- **OpenCV**: Face detection (`cv2`).
- **Matplotlib**: Plots (`plt`).
- **Pathlib**: File paths (`Path`).
- **Tarfile**: Extraction.
- **DeepFace**: Gender prediction.
- **MediaPipe**: Hair segmentation (`mp`).
- **Scikit-learn**: Clustering (`KMeans`).

## Dataset
- Source: `lfw-funneled.tgz` (LFW dataset).
- Size: 5,749 individuals.
- Structure: Person directories with JPG images.
- Sample: 5 images processed.

## Key Features
- **Accurate Detection**: Haar Cascade for face localization.
- **Advanced Prediction**: DeepFace and MediaPipe for attributes.
- **Visual Output**: Images with bounding boxes and labels.
- **Kaggle-Friendly**: Runs in Kaggle environment.

## Installation
```bash
pip install opencv-python matplotlib deepface mediapipe scikit-learn
