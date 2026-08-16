# CNN Image Classification - SVHN

## Overview
A CNN-based digit-classification project using the SVHN (Street View House Numbers) dataset.

The project keeps the same main CNN idea while using a different dataset. It also includes detailed evaluation, filter visualization, model saving, and fine-tuning.

## Dataset
SVHN contains real-world street-view digit images.

- RGB images
- Image size: `32x32x3`
- 10 classes: digits `0` to `9`

The dataset is downloaded as `.mat` files and loaded with SciPy.

## Preprocessing
- Convert the dataset from `(32, 32, 3, N)` to `(N, 32, 32, 3)`
- Normalize pixel values to `[0, 1]`
- Convert SVHN label `10` to digit `0`
- One-hot encode the labels

## CNN
The model contains:
- Convolutional layers
- Max Pooling
- Batch Normalization
- Flatten
- Dense layers
- Dropout
- 10-class output layer

The optimizer is RMSprop.

## Evaluation
The notebook includes:
- Test Loss
- Test Accuracy
- Accuracy and Loss curves
- Classification Report
- Confusion Matrix
- Balanced Accuracy
- BER
- First-layer CNN filter visualization

## Results

### Before Fine-Tuning
```text
Test Accuracy: 89.87%
Test Loss:     0.0626
```

### After Fine-Tuning
The model was fine-tuned with a lower learning rate of `0.0001`.

```text
Test Accuracy: 91.05%
Test Loss:     0.0547
```

Improvement:
```text
+1.18 percentage points
```

## Model Saving
```text
svhn_cnn.keras
svhn_cnn_finetuned.keras
```

Weights can also be saved separately as `.weights.h5`.

## How to Run
1. Open `CNN_SVHN_Complete_Colab.ipynb` in Google Colab.
2. Run the cells in order.
3. Train the CNN.
4. Evaluate the initial model.
5. Run the fine-tuning stage.
6. Save and download the best model.

## Requirements
Google Colab, Python 3, TensorFlow, NumPy, SciPy, Matplotlib, scikit-learn, and Seaborn.
