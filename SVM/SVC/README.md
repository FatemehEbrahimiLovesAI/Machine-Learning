# SVC Classification Example

This Jupyter notebook demonstrates a simple binary classification task using Support Vector Classifier (SVC) with a linear kernel on synthetic data.

## Overview

The notebook generates synthetic 2D data and trains a linear SVM model to classify it into two classes. It then evaluates the model's accuracy on a test set.

## Code Structure

### 1. Import Libraries
```python
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
```

### 2. Generate Synthetic Dataset
- **n_samples**: 200 data points
- **n_features**: 2 features (for easy visualization)
- **n_informative**: 2 informative features
- **n_redundant**: 0 redundant features
- **n_classes**: 2 classes (binary classification)
- **random_state**: 12 (for reproducibility)

### 3. Train-Test Split
- **Training set**: 80% of data (160 samples)
- **Test set**: 20% of data (40 samples)
- **random_state**: 12 (ensures consistent split)

### 4. Model Training
- **Algorithm**: Support Vector Classifier (SVC)
- **Kernel**: Linear
- The model is trained on the training data using `fit()`

### 5. Evaluation
- Predictions are made on the test set
- Accuracy score is calculated by comparing predictions with actual labels

## Results

The model achieves an accuracy of **0.95** (95%) on the test set, indicating excellent performance on this synthetic dataset.

## Requirements

- scikit-learn
- Python 3.x

## Usage

Run all cells sequentially in Jupyter Notebook or JupyterLab to reproduce the results.
