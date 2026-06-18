# California Housing Price Prediction with Mini-Batch Gradient Descent

This project implements a multivariate linear regression model from scratch using NumPy to predict housing prices in California. The model is trained with mini-batch stochastic gradient descent (SGD) without relying on any machine learning framework for the core learning algorithm.

## Overview

The notebook walks through a full supervised learning pipeline: loading a real-world dataset, preprocessing features, implementing regression and gradient descent by hand, and training the model over multiple epochs.

## Dataset

The project uses the California Housing dataset from scikit-learn (`fetch_california_housing`). It contains 20,640 samples with 8 numerical features describing various housing and demographic attributes of California districts.

Features:

- MedInc — median income in block group
- HouseAge — median house age in block group
- AveRooms — average number of rooms per household
- AveBedrms — average number of bedrooms per household
- Population — block group population
- AveOccup — average number of household members
- Latitude — block group latitude
- Longitude — block group longitude

Target: median house value for California districts (in units of $100,000).

## Data Preprocessing

Two preprocessing steps are applied to the feature matrix:

1. Missing value imputation: zero values in each feature are treated as missing and replaced with the column mean.
2. Min-max normalization: each feature is scaled to the range [0, 1] using the formula `(x - min) / (max - min)`.

## Model

The model is a standard multivariate linear regression:

```
y_hat = X @ w
```

where `X` is the feature matrix and `w` is the weight vector, initialized randomly from a standard normal distribution.

## Training

The model is trained using mini-batch gradient descent with the following components:

- Loss function: Mean Squared Error (MSE)
- Gradient computation: analytical gradient of MSE with respect to each weight
- Weight update rule: `w = w - eta * grad`
- Batch size: 8
- Epochs: 100
- Learning rate (eta): 0.1

The dataset is divided into 2,580 mini-batches per epoch. At the end of each epoch, the average MSE across all batches is printed.

## Project Structure

```
main.ipynb    — single notebook containing all code: data loading, preprocessing,
                model definition, training loop, and evaluation utilities
```

## Requirements

- Python 3.x
- NumPy
- Pandas
- scikit-learn (for dataset loading only)

Install dependencies:

```bash
pip install numpy pandas scikit-learn
```

## Usage

Open and run the notebook top to bottom:

```bash
jupyter notebook main.ipynb
```

All cells are self-contained and must be executed in order. Training progress is printed to stdout after each epoch in the format:

```
epoch : 1 mse: <value>
epoch : 2 mse: <value>
...
```

## Notes

- The regression and gradient descent logic are implemented entirely with NumPy — no scikit-learn estimators are used for training.
- This implementation is intended for educational purposes and does not include train/test splitting, bias term, regularization, or convergence checks.
- Adding a column of ones to `X` before training would introduce a bias term and is a natural next step.

---

## Roadmap

This project is actively under development. The following improvements and experiments are planned for upcoming versions.

### Hyperparameter Experimentation

The model will be systematically tested across a range of hyperparameter configurations to study their effect on convergence and final performance:

- Learning rates: multiple values will be compared (e.g., 0.001, 0.01, 0.1, 0.5) to observe underfitting, stable convergence, and divergence behavior.
- Epochs: training runs with varying numbers of epochs will be conducted to analyze learning curves and identify the point of diminishing returns.
- Mini-batch sizes: different batch sizes (e.g., 1, 8, 32, 64, 256) will be tested to compare the noise and speed trade-offs of SGD, mini-batch GD, and full-batch GD.

Results from these experiments will be logged and visualized for side-by-side comparison.

### Train / Test Split

The dataset will be split into training and test sets (e.g., 80/20) before any preprocessing or model fitting. Preprocessing statistics (mean, min, max) will be computed on the training set only and then applied to the test set to prevent data leakage. The model will be re-trained on the training split and evaluated on the held-out test set to obtain an unbiased estimate of generalization performance.

### Additional Evaluation Metrics

Beyond MSE, the following metrics will be added to give a more complete picture of model quality:

- Root Mean Squared Error (RMSE) — to express error in the same unit as the target variable.
- Mean Absolute Error (MAE) — to measure average absolute deviation, less sensitive to outliers than MSE.
- R-squared (R2) — to measure the proportion of variance in the target explained by the model.

### Visualization with Matplotlib

A dedicated visualization section will be added to the notebook covering:

- Loss curves: training MSE plotted over epochs for each hyperparameter configuration.
- Hyperparameter comparison plots: side-by-side line charts comparing convergence behavior across different learning rates, batch sizes, and epoch counts.
- Predicted vs. actual scatter plot: to visually assess model fit on the test set.
- Residual distribution: histogram of prediction errors to check for systematic bias.

### Documentation

All functions will be documented with NumPy-style docstrings describing parameters, return values, and behavior. Markdown cells will be added throughout the notebook to explain each section, the reasoning behind design decisions, and the interpretation of results.

### Dependency addition

```
matplotlib
```

Full updated install command:

```bash
pip install numpy pandas scikit-learn matplotlib
```
