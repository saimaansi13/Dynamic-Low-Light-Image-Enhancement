## Dynamic-Low-Light-Image-Enhancement
## Overview

This project focuses on enhancing low-light images and developing a denoising model using convolutional neural networks (CNN). The pipeline involves preprocessing low-light images using computer vision techniques, augmenting the dataset, training a denoising model, and evaluating its performance on new images.

## Techniques Used

### 1. Contrast Limited Adaptive Histogram Equalization (CLAHE)
CLAHE is a variant of adaptive histogram equalization (AHE) that enhances local contrast by limiting the amplification of the contrast in homogeneous regions. In this project, CLAHE is applied to each color channel of the low-light image to improve visibility while preserving details.
(```clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_b = clahe.apply(b)
clahe_g = clahe.apply(g)
clahe_r = clahe.apply(r)
clahe_color_image = cv2.merge([clahe_b, clahe_g, clahe_r])```)


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
