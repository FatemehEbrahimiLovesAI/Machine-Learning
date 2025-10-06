# Wine Dataset Classification Project

## Overview

This project demonstrates the use of two supervised machine learning algorithms — K-Nearest Neighbors (KNN) and Logistic Regression — to classify different types of wines based on their chemical features.

Both models were trained and evaluated on the Wine dataset from the scikit-learn library.
Before training, the dataset was standardized using StandardScaler to ensure that all features had a similar scale and range.


---

## Dataset Information

Source: sklearn.datasets.load_wine()

Number of samples: 178

Number of features: 13 (chemical properties such as alcohol, malic acid, ash, etc.)

Target classes: 3 (types of wine)



---

## Preprocessing Steps

1. Load the dataset using load_wine()


2. Split data into training and testing sets using train_test_split


3. Standardize features using StandardScaler

This step scales all numeric features to have mean = 0 and variance = 1

Important for algorithms that rely on distance (like KNN)





---

## Models Used

### Logistic Regression

A linear model that predicts the probability of a sample belonging to each class.

Great for datasets with linearly separable classes.

Accuracy achieved: 100% on the test set 


### K-Nearest Neighbors (KNN)

A non-parametric algorithm that classifies a sample based on its closest neighbors.

Works well on non-linear data but can be slower with large datasets.

Accuracy achieved: 97% on the test set ⚡



---

## Results Comparison

Model	Accuracy (%)

Logistic Regression	 : 100
KNN	: 97


- Logistic Regression performed slightly better, likely because the dataset has linearly separable classes.
- Both models achieved high accuracy after scaling, showing the importance of feature normalization.


---

## Key Insights

Always normalize your data before training distance-based models like KNN.

Logistic Regression can outperform more complex models when the data is clean and linear.

Comparing multiple models using cross-validation or test sets is essential for selecting the best approach.



---

## How to Run

1. **Install dependencies:**

pip install scikit-learn pandas numpy


2. **Run the notebook:**

jupyter notebook wine_dataset.ipynb


3. **View the accuracy comparison and plots in the output cells.**




---

## Conclusion

This experiment shows that model performance heavily depends on the nature of the dataset.
Even simple models like Logistic Regression can achieve perfect accuracy when the data is well-structured and preprocessed correctly.
