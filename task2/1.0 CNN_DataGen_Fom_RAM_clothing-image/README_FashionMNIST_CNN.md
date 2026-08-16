# Fashion-MNIST CNN

## Overview

This notebook uses the **same Fashion-MNIST dataset and CNN experiments** from the original notebook, with a few cleanup changes and model-saving cells added at the end.

## Dataset

Fashion-MNIST:
- 60,000 training images
- 10,000 test images
- Grayscale images: `28 × 28`
- 10 classes

Classes:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

The dataset is loaded automatically with TensorFlow/Keras.

## Main CNN

The main CNN contains:
- Conv2D: 32 filters, `3×3`
- MaxPooling
- Conv2D: 64 filters, `3×3`
- MaxPooling
- Conv2D: 128 filters, `3×3`
- Flatten
- Dense: 64 units
- Dense: 10 units with Softmax

Optimizer: Adam  
Loss: Sparse Categorical Cross-Entropy

## Experiments

### 1. Main CNN
The main model uses the full Fashion-MNIST training set with a 15% validation split.

### 2. Small-data CNN
A smaller CNN is trained on the first 500 training images.

### 3. Data Augmentation
The small-data model is cloned and trained with:
- `height_shift_range=3`
- `horizontal_flip=True`

The test set is kept for final evaluation.

## Evaluation

The notebook includes:
- Test accuracy
- Test loss
- Training/validation accuracy curves
- Training/validation loss curves
- Sample prediction visualization
- First convolution-layer filter visualization

## Saved Models

At the end, the notebook saves:

```text
fashion_mnist_main_cnn.keras
fashion_mnist_small_cnn.keras
fashion_mnist_augmented_cnn.keras
```

Optional weights backups:

```text
fashion_mnist_main_cnn.weights.h5
fashion_mnist_small_cnn.weights.h5
fashion_mnist_augmented_cnn.weights.h5
```

The final augmented model is also downloaded directly from Google Colab.

## How to Run

1. Open `FashionMNIST_CNN_Colab_ready.ipynb` in Google Colab.
2. Run all cells in order.
3. Review the evaluation and visualization results.
4. Run the final saving cells.
5. Download the model you want to keep.

## Notes

The dataset and overall experiments were kept the same as the supplied notebook. The main cleanup was to make the training history explicit, keep the test set for final evaluation in the first experiments, and add reliable model/weights saving at the end.
