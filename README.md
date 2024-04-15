## Dynamic-Low-Light-Image-Enhancement
## Overview

This project focuses on enhancing low-light images and developing a denoising model using convolutional neural networks (CNN). The pipeline involves preprocessing low-light images using computer vision techniques, augmenting the dataset, training a denoising model, and evaluating its performance on new images.

## Image Pre-Processing Pipeline
### Original Image 

<img width="404" alt="Original Image " src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/43187aa1-d98c-4955-9fb8-957a42be68a9">

### LAB Color Space Conversion
The LAB color space comprises three components: L (lightness), A (green-red), and B (blue-yellow). Converting the original image to LAB allows for the separation of luminance from chrominance, enabling independent manipulation of brightness and color information. By isolating and enhancing the luminance component (L-channel), it becomes possible to improve brightness and clarity without significantly amplifying noise or introducing unwanted artifacts. Moreover, preserving color accuracy and detail through the separation of luminance and chrominance results in more natural-looking and visually appealing low-light image conversions.
```python
lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
```

<img width="404" alt="After LAB" src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/94b2c6dc-dfeb-4fe5-806e-fcf69a01eae4">

### Contrast Limited Adaptive Histogram Equalization (CLAHE)
Contrast Limited Adaptive Histogram Equalization (CLAHE) is a variant of histogram equalization that enhances local contrast in an image while limiting the amplification of noise. CLAHE divides the image into tiles and applies histogram equalization to each tile separately, preventing over-amplification of intensity variations. This resulted in improved contrast and enhanced details, particularly in regions with low contrast.
```python
# Apply CLAHE to the L channel of the LAB image
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l, a, b = cv2.split(lab_image)
l_clahe = clahe.apply(l)
clahe_lab_image = cv2.merge([l_clahe, a, b])
```
<img width="406" alt="After CLAHE" src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/8177f01a-31d6-4e23-9780-b9ca3c0abd6d">

### Linear Brightening and RGB Conversion
The CLAHE-enhanced LAB image is first converted to RGB color space.Following the conversion, linear brightening is applied to the RGB image. Here each pixel's value is multiplied by a specified factor, adjusting the overall brightness while preserving the relative contrast. Finally, the resulting brightened image is normalized to range between 0 and 1, preparing it for subsequent processing steps.
```python
clahe_color_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
brightness_adjusted_image = np.clip(clahe_color_image * brightness_factor, 0, 255).astype(np.uint8)
preprocessed_image = brightness_adjusted_image / 255.0
```
<img width="524" alt="After Linear brightning" src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/8949228b-9715-407b-b128-cd9db8dbd7dd">

### Data Augmentation with Albumentations
The albumentations library is used to perform data augmentation on images. The transform variable contains a composition of augmentation techniques, including Gaussian noise addition (GaussNoise) and random brightness and contrast adjustments (RandomBrightnessContrast).The num_augmentations variable specifies the number of augmented images to generate per original image during training. These transformations helped introduce variations in brightness, contrast, and noise levels, making the dataset more robust and diverse. 

```python
import albumentations as A
transform = A.Compose([
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.2, 0.2), p=1.0),
])
num_augmentations = 2
```
![Augument 1](https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/6bb25721-444d-40ae-8c03-c4b1fb4c760c)  ![Augument 2](https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/41642b41-fbe6-40c0-a71f-de6ba49dbb99)

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
