### Diabetes Prediction using K-Nearest Neighbors (KNN)

This project is a simple machine learning pipeline that uses the K-Nearest Neighbors (KNN) algorithm to predict diabetes based on medical diagnostic measurements.

## Features

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age


## Preprocessing

Missing Value Handling:
Some features contain 0 values which are not realistic (e.g., 0 for glucose or blood pressure). These are replaced by the mean value of each respective column.

Feature Scaling:
Features are standardized using StandardScaler from scikit-learn to improve model performance.


## Model

Algorithm: K-Nearest Neighbors (KNN)

Number of Neighbors: 33

Train/Test Split: 80% training, 20% testing


## Results

Accuracy: 81%


## Requirements

Python 3.x

NumPy

Pandas

Matplotlib

scikit-learn


You can install dependencies using:

pip install numpy pandas matplotlib scikit-learn

## How to Run

1. Make sure the dataset file diabetes.csv is in the correct path.


2. Run the script:



python knn_diabetes.py

3. The script will print the accuracy score of the model.



## File Structure

.
├── diabetes.csv
└── knn_diabetes.py

## Notes

The number of neighbors (n_neighbors=33) was chosen manually and may be further optimized using hyperparameter tuning.

For better results, consider using cross-validation or experimenting with other algorithms like Random Forest or Logistic Regression.
