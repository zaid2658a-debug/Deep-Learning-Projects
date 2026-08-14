# Deep Learning Labs — Fully Connected Neural Networks (FCNN)

This folder contains hands-on deep learning labs built with **TensorFlow / Keras**.
Each notebook takes a real-world tabular dataset through the full ML pipeline:
data exploration → cleaning/feature engineering → preprocessing → building an
FCNN → training → evaluation.

## 📓 Notebooks

| Notebook | Task | Dataset | Key Result |
|---|---|---|---|
| [`Churn_Modelling_FCNN.ipynb`](./Churn_Modelling_FCNN.ipynb) | Binary Classification | `Churn_Modelling.csv` | Accuracy **86%**, ROC-AUC **0.86** |
| [`Bengaluru_House_Price_FCNN.ipynb`](./Bengaluru_House_Price_FCNN.ipynb) | Regression | `bengaluru_house_prices.csv` | R² **0.68**, MAE **≈33.4 lakhs** |
| `2. FCNN_Concrete-project.ipynb` | Regression | Concrete compressive strength | FCNN lab |
| `3. fashion_mnist_fcc.ipynb` | Multi-class Classification | Fashion MNIST | FCNN lab |

## 1. Customer Churn Prediction

**Goal:** Predict whether a bank customer will leave the bank (`Exited` = 1) based on
demographic and account features (credit score, geography, gender, age, tenure,
balance, number of products, credit card ownership, active-member status, estimated
salary).

**Approach:**
- Dropped non-predictive identifier columns (`RowNumber`, `CustomerId`, `Surname`)
- One-hot encoded `Geography`, label-encoded `Gender`
- Standardized numeric features
- FCNN: `Dense(32) → Dropout → Dense(16) → Dropout → Dense(8) → Dense(1, sigmoid)`
- Trained with `EarlyStopping` on validation loss

**Results:** 86% test accuracy, ROC-AUC of 0.86. Confusion matrix, ROC curve, and
training curves are included in the notebook.

## 2. Bengaluru House Price Prediction

**Goal:** Predict house `price` (in lakhs ₹) from property features (location, area
type, total square footage, bedrooms, bathrooms, balconies).

**Approach:**
- Parsed messy `total_sqft` values (ranges like `"2100-2850"` averaged)
- Extracted number of bedrooms (`bhk`) from the `size` column
- Grouped rare `location` values into `"other"` to reduce cardinality
- Removed data-entry outliers (unrealistic sqft-per-bedroom, extreme price/sqft)
- One-hot encoded categorical features, standardized numeric features and target
- FCNN: `Dense(128) → Dropout → Dense(64) → Dropout → Dense(32) → Dense(1, linear)`
- Trained with `EarlyStopping` on validation loss

**Results:** R² of 0.68, MAE of ~33.4 lakhs on the test set. Actual-vs-predicted
scatter plot and training curves are included in the notebook.

## 🛠 Requirements

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

## ▶️ How to run

1. Clone the repo and open the notebook of interest in Jupyter / Colab / VS Code.
2. Make sure the corresponding CSV file (`Churn_Modelling.csv` or
   `bengaluru_house_prices.csv`) is in the same folder as the notebook.
3. Run all cells top to bottom.

