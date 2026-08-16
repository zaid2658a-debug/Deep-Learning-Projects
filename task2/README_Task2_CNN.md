# Task 2 - CNN Projects

This folder contains the CNN experiments developed for Task 2.

## Projects

### 0.0 FashionMNIST_Cnn

A Convolutional Neural Network project using the **Fashion-MNIST** dataset.

The project includes:

- Dataset loading and preprocessing
- CNN model training
- Accuracy and loss visualization
- Test-set evaluation
- Classification and visualization experiments
- Model saving

### 0.1 SVHN_Cnn

A CNN project using the **SVHN (Street View House Numbers)** dataset.

The project includes:

- SVHN dataset loading and preprocessing
- CNN training
- Accuracy and loss curves
- Test-set evaluation
- Classification report
- Confusion matrix
- BER / balanced-accuracy evaluation
- CNN filter visualization
- Model saving
- Fine-tuning with a lower learning rate

The latest fine-tuning result achieved approximately:

```text
Test Accuracy: 91.05%
Test Loss:     0.0547
```

### 1.0 CNN_DataGen_From_RAM_clothing-image

This folder contains the additional CNN/Data Generator experiment for clothing-image data.

## Common Workflow

The CNN projects generally follow this workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
CNN Model
   ↓
Training
   ↓
Validation
   ↓
Evaluation
   ↓
Visualization
   ↓
Model Saving
```

## Model Files

The notebooks include cells for saving the trained models in Keras format (`.keras`) and, where needed, saving model weights separately.

## Running the Notebooks

1. Open the required notebook in Google Colab.
2. Run the cells in order.
3. Download or load the dataset as required by that project.
4. Train the CNN.
5. Review the evaluation results and visualizations.
6. Save the trained model using the final cells.

## Main Libraries

- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- SciPy
- Seaborn

## Folder Structure

```text
task2_Cnn/
├── 0.0 FashionMNIST_Cnn/
├── 0.1 SVHN_Cnn/
├── 1.0 CNN_DataGen_From_RAM_clothing-image/
└── README.md
```
