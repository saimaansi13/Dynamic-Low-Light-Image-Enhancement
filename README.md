## Dynamic-Low-Light-Image-Enhancement
## Overview

This project focuses on enhancing low-light images and developing a denoising model using convolutional neural networks (CNN). The pipeline involves preprocessing low-light images using computer vision techniques, augmenting the dataset, training a denoising model, and evaluating its performance on new images.

## Image Processing Pipeline
### Original Image 
![Original Image](https://ibb.co/yqY3Kc1)

### LAB Color Space Conversion
The LAB color space comprises three components: L (lightness), A (green-red), and B (blue-yellow). Converting the original image to LAB allows for the separation of luminance from chrominance, enabling independent manipulation of brightness and color information. By isolating and enhancing the luminance component (L-channel), it becomes possible to improve brightness and clarity without significantly amplifying noise or introducing unwanted artifacts. Moreover, preserving color accuracy and detail through the separation of luminance and chrominance results in more natural-looking and visually appealing low-light image conversions.

### 2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
CLAHE is a variant of adaptive histogram equalization (AHE) that enhances local contrast by limiting the amplification of the contrast in homogeneous regions. In this project, CLAHE is applied to each color channel of the low-light image to improve visibility while preserving details.
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_b = clahe.apply(b)
clahe_g = clahe.apply(g)
clahe_r = clahe.apply(r)
clahe_color_image = cv2.merge([clahe_b, clahe_g, clahe_r])
```


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
