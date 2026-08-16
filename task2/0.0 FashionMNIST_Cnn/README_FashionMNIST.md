# CNN Image Classification - Fashion-MNIST

## Overview
A CNN image-classification project using the Fashion-MNIST dataset in Google Colab.

## Dataset
Fashion-MNIST contains 70,000 grayscale images of size `28x28x1` across 10 classes:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

The dataset is downloaded automatically using TensorFlow/Keras.

## Preprocessing
- Normalize pixel values to `[0, 1]`
- Add the channel dimension for CNN input
- Convert labels to one-hot encoding

## Workflow
- Load and preprocess Fashion-MNIST
- Build and train the CNN
- Plot Accuracy and Loss curves
- Evaluate the model
- Generate a Classification Report
- Generate a Confusion Matrix
- Save the model and weights

## Model Saving
```text
cnn_fashion_mnist.keras
cnn_fashion_mnist.weights.h5
```

## How to Run
1. Open `CNN_FashionMNIST_Colab_ready.ipynb` in Google Colab.
2. Run the cells in order.
3. Review the evaluation results.
4. Save and download the model.

## Requirements
Google Colab, Python 3, TensorFlow, NumPy, Matplotlib, scikit-learn, and Seaborn.
