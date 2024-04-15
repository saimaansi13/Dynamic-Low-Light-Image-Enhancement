# Dynamic-Low-Light-Image-Enhancement

## Overview

This project focuses on enhancing low-light images and developing a denoising model using convolutional neural networks (CNN). The pipeline involves preprocessing low-light images using computer vision techniques, augmenting the dataset, training a denoising model, and evaluating its performance on new images.

## Image Pre-Processing Pipeline
The preprocessing pipeline involves several steps to prepare low-light images for denoising:

### Original Image 

<img width="404" alt="Original Image " src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/43187aa1-d98c-4955-9fb8-957a42be68a9">

### LAB Color Space Conversion
The LAB color space comprises three components: L (lightness), A (green-red), and B (blue-yellow). Converting the original image to LAB allows for the separation of luminance from chrominance, enabling independent manipulation of brightness and color information. By isolating and enhancing the luminance component (L-channel), it becomes possible to improve brightness and clarity without significantly amplifying noise or introducing unwanted artifacts. Moreover, preserving color accuracy and detail through the separation of luminance and chrominance results in more natural-looking and visually appealing low-light image conversions.
```python
lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
```

<img width="406" alt="labimgg" src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/f88de295-e223-49de-be88-7bb56b3497d4">


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
The CLAHE-enhanced LAB image is first converted to RGB color space. Following the conversion, linear brightening is applied to the RGB image. Here each pixel's value is multiplied by a specified factor, adjusting the overall brightness while preserving the relative contrast. Finally, the resulting brightened image is normalized to range between 0 and 1, preparing it for subsequent processing steps.
```python
clahe_color_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
brightness_adjusted_image = np.clip(clahe_color_image * brightness_factor, 0, 255).astype(np.uint8)
preprocessed_image = brightness_adjusted_image / 255.0
```
<img width="406" alt="After Linear brightning" src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/8949228b-9715-407b-b128-cd9db8dbd7dd">

## Data Augmentation with Albumentations
The albumentations library is used to perform data augmentation on images. The transform variable contains a composition of augmentation techniques, including Gaussian noise addition (GaussNoise) and random brightness and contrast adjustments (RandomBrightnessContrast).The num_augmentations variable specifies the number of augmented images to generate per original image during training. These transformations were applied to the preprocessed low-light images to introduce variations in brightness, contrast, and noise levels, enhancing the dataset's robustness and diversity. 

```python
import albumentations as A
transform = A.Compose([
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.2, 0.2), p=1.0),
])
num_augmentations = 2
```

<img width="428" alt="aug1" src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/12982f77-efc7-4ee3-bb49-447b040e40aa">
<img width="427" alt="aug2" src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/3d19055e-a6ad-4fe5-a3c4-5263dc25d8f9">


## Image Denoising Model

I designed and trained a custom image denoising model using convolutional neural network (CNN) layers. The model architecture includes multiple convolutional layers with rectified linear unit (ReLU) activation functions to capture complex features and reduce noise in the input images. The final layer utilizes linear activation to generate denoised images with three channels corresponding to RGB color channels.

To optimize the model's performance, I chose to train the model with the Adam optimizer and MSE loss function. The model was trained on a dataset consisting of noisy preprocessed images, including both the original low-light images and their corresponding augmentations. Additionally, I augmented the dataset by adding pairs of augmented images along with their corresponding high-quality images. This augmentation was done to provide the model with a diverse set of training examples, enhancing its ability to generalize to unseen data.

During training, the model's performance was evaluated on a separate validation dataset. The training process involved iterating over the dataset for a specified number of epochs with a batch size of 32.

<img width="1000" alt="Training img " src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/0ce0f329-276a-4396-ab8b-3612cd1c6fcb">

```python
model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(None, None, 3)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(3, (3, 3), activation='linear', padding='same')
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
```
## Example Image
### Input Image: Raw Low-Light Image
<img width="613" alt="input_img" src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/38435584-658a-4b30-8b97-3b3b721cfddc">

### Final Output Image: Enhanced and denoised image
<img width="610" alt="Final output_img" src="https://github.com/saimaansi13/Dynamic-Low-Light-Image-Enhancement/assets/125540201/b7a2d327-6214-4404-9601-c1e43e86fb99">

## Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- Albumentations
- Computational resources for model training and evaluatio
- Other dependencies

## Dataset - LOL Dataset - https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
