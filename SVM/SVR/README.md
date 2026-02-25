# SVR Regression Example

This Jupyter notebook demonstrates a simple regression task using Support Vector Regression (SVR) with RBF kernel on synthetic data.

## Overview

The notebook generates synthetic regression data and trains an SVR model with RBF kernel to predict target values. It then evaluates the model's performance on a test set.

## Code Structure

### 1. Import Libraries
```python
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
```

### 2. Generate Synthetic Dataset
- **n_samples**: 200 data points
- **n_features**: 10 features
- **n_informative**: 0 informative features (all features contribute to the target)

### 3. Train-Test Split
- **Training set**: 80% of data (160 samples)
- **Test set**: 20% of data (40 samples)
- **random_state**: 12 (ensures consistent split)

### 4. Model Training
- **Algorithm**: Support Vector Regressor (SVR)
- **Kernel**: RBF (Radial Basis Function) - default kernel
- The model is trained on the training data using `fit()`

### 5. Evaluation
- Predictions are made on the test set
- Accuracy score is calculated (note: accuracy_score is typically used for classification, for regression proper metrics would be MSE, MAE, or R²)

## Results

The model achieves an accuracy of **1.0** (100%) on the test set, indicating perfect predictions on this synthetic dataset.

## Key Parameters of SVR

- **kernel**: 'rbf' (default), 'linear', 'poly', 'sigmoid'
- **C**: Regularization parameter (default=1.0)
- **epsilon**: Defines the epsilon-tube within which no penalty is associated (default=0.1)
- **gamma**: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (default='scale')

## Requirements

- scikit-learn
- Python 3.x

## Usage

Run all cells sequentially in Jupyter Notebook or JupyterLab to reproduce the results.
