# Handwritten Digit Recognition

## Overview
This project implements a **Handwritten Digit Recognition** system using **Deep Learning**. The model is trained on the **MNIST dataset**, which consists of grayscale images of handwritten digits (0-9). The goal is to accurately classify handwritten digits using a neural network.

## Features
- Preprocessing of handwritten digit images
- Neural network model built using **TensorFlow/Keras**
- Training and evaluation on the **MNIST dataset**
- Visualization of training progress and predictions

## Dataset
The project uses the **MNIST dataset**, which contains:
- **60,000** training images
- **10,000** test images

Each image is **28x28 pixels** in grayscale format.

## Installation & Setup
To run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Laksha10/Handwritten-Digit-Recognition.git
   cd Handwritten-Digit-Recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook handwritten-digit-recognition.ipynb
   ```

## Model Architecture
The model is a **Convolutional Neural Network (CNN)** with the following layers:
- **Convolutional layers** for feature extraction
- **Max-pooling layers** for dimensionality reduction
- **Fully connected layers** for classification
- **Softmax activation** for predicting digit probabilities

## Results
After training, the model achieves an accuracy of approximately **98%** on the test dataset.


