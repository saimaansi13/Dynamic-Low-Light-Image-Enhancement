## Dynamic-Low-Light-Image-Enhancement
## Overview

This project focuses on enhancing low-light images and developing a denoising model using convolutional neural networks (CNN). The pipeline involves preprocessing low-light images, augmenting the dataset, training a denoising model, and evaluating its performance on new images.

## Project Structure

### 1. Preprocess Low-Light Images
- **Input:** Raw low-light images in 'our485' dataset.
- **Output:** Preprocessed low-light images using Contrast Limited Adaptive Histogram Equalization (CLAHE) and linear brightening.

### 2. Augment Low-Light Images
- **Input:** Preprocessed low-light images and corresponding high-quality images.
- **Output:** Augmented datasets for training the denoising model.

### 3. Train Denoising Model
- **Architecture:** A CNN with convolutional layers for noise reduction and brightness enhancement.
- **Training:** Trained on augmented low-light images and their high-quality counterparts.

### 4. Evaluate Denoising Model
- **Input:** New low-light images for evaluation.
- **Output:** Denoised images using the pre-trained model.

### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- Albumentations
- Other dependencies

### Dataset - LOL Dataset - https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
