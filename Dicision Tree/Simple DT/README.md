# Decision Tree Classifier

This notebook demonstrates the implementation of a Decision Tree classifier for binary classification using scikit-learn.

## Overview

The notebook builds and evaluates a Decision Tree model on a synthetic dataset, with a focus on controlling model complexity through the `max_depth` parameter.

## Dataset

The synthetic dataset is generated using `make_classification` with the following parameters:
- 1000 samples
- 10 features
- 2 informative features
- 2 classes

The dataset is split into training (80%) and testing (20%) sets with a fixed random state (12) for reproducibility.

## Model Configuration

- **Algorithm**: Decision Tree Classifier
- **Max Depth**: 5 (to prevent overfitting)
- **Other parameters**: Default values (Gini impurity, best split strategy)

## Results

- **Test Accuracy**: 0.935 (93.5%)

The model achieved strong performance with a depth-limited tree, indicating good generalization without overfitting.

## Key Features

1. **Controlled Tree Depth**: Limited to depth 5 to balance model complexity and performance
2. **Visualization Ready**: The notebook imports `plot_tree` for potential tree visualization
3. **Reproducible Results**: Fixed random state ensures consistent train-test splits

## Dependencies

- scikit-learn
- matplotlib

## Usage

Run the notebook cells sequentially to:
1. Generate the synthetic dataset
2. Split the data into training and testing sets
3. Train a Decision Tree classifier with max_depth=5
4. Evaluate model accuracy on the test set
