# SVM Kernels Comparison

This notebook demonstrates the implementation and comparison of different kernel functions for Support Vector Machine (SVM) classification using scikit-learn.

## Overview

The notebook explores four different SVM kernels on a synthetic binary classification dataset:
- Linear Kernel
- Polynomial Kernel (Poly)
- Sigmoid Kernel
- Radial Basis Function Kernel (RBF)

## Dataset

The synthetic dataset is generated using `make_classification` with the following parameters:
- 200 samples
- 20 features
- 2 informative features
- 0 redundant features
- 2 classes
- Random state: 12

The dataset is split into training (80%) and testing (20%) sets.

## Kernels Tested

### 1. Linear Kernel
- **Accuracy**: 0.90 (90%)
- Simple linear decision boundary
- Best for linearly separable data

### 2. Polynomial Kernel
- **Accuracy**: 0.80 (80%)
- Uses polynomial combinations of features
- Degree parameter set to default (3)

### 3. Sigmoid Kernel
- **Accuracy**: 0.925 (92.5%)
- Similar to a neural network activation function
- Good performance on this dataset

### 4. RBF Kernel
- **Accuracy**: 0.925 (92.5%)
- Most flexible kernel, can create complex decision boundaries
- Tied for best performance with sigmoid kernel

## Results Summary

| Kernel | Accuracy |
|--------|----------|
| Linear | 0.900 |
| Polynomial | 0.800 |
| Sigmoid | 0.925 |
| RBF | 0.925 |

## Key Takeaways

1. **RBF and Sigmoid** kernels performed best on this dataset with 92.5% accuracy
2. **Linear kernel** performed reasonably well (90%), suggesting the data has some linear separability
3. **Polynomial kernel** underperformed compared to others, possibly due to default parameters not suiting this dataset
4. All models used default parameters except for the kernel specification

## Dependencies

- scikit-learn
- matplotlib
- numpy

## Usage

Run the notebook cells sequentially to:
1. Generate the synthetic dataset
2. Train SVM models with different kernels
3. Compare their accuracies
