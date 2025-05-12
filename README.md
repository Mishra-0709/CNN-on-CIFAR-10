# 🧠 Image Classification on CIFAR-10 using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**, a standard benchmark in computer vision.

## 🚀 Project Overview

The CIFAR-10 dataset consists of 60,000 32×32 color images in 10 classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

This project builds and trains a CNN model using **TensorFlow** to classify these images.

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## 📁 Dataset

- **Name**: CIFAR-10
- **Source**: Built-in TensorFlow dataset (`tensorflow.keras.datasets.cifar10`)
- **Size**: 60,000 images (50,000 train, 10,000 test)
- **Image Shape**: 32x32 RGB
🏗️ Model Architecture
Conv2D (32 filters) + ReLU + MaxPooling
Conv2D (64 filters) + ReLU + MaxPooling
Flatten → Dense (128 units) + ReLU + Dropout
Output Layer: Dense (10 units, softmax)
![image](https://github.com/user-attachments/assets/60185710-d18d-47b9-9791-45ffc68c0049)

```python
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
