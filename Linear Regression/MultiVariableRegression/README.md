### Linear Regression from Scratch (Boston Housing Example)

## Overview
This project implements Multivariable Linear Regression from scratch using only NumPy and pandas, along with standard preprocessing tools from scikit-learn.
It includes two main scripts:

**train.py** → trains a linear regression model on the dataset.

**test.py** → evaluates the trained model on the dataset using saved weights and scaler.



---

## Files Overview

**train.py**

* Loads the dataset (BostonHousing.csv).

* Splits features (X) and target (y).

* Normalizes features using StandardScaler and saves the fitted scaler (scaler.pkl) for later use.

* Adds a bias (intercept) column to the input data.

* Initializes random weights.

***Implements:***

- Mean Squared Error (MSE) function

- Linear Regression prediction function

- Gradient calculation

- Gradient Descent update rule


Trains the model for 100 epochs while printing the loss and weights.

Saves the trained weights into MultiVariablelinearRegressionWeights.csv.npy.


**test.py**

* Loads the saved weights and scaler from training.

* Loads the dataset (BostonHousing.csv).

* Splits features (X) and target (y).

* Applies the same normalization (using the saved scaler).

***Implements:***

* Linear Regression prediction function

* Mean Squared Error (MSE) function

* R² Score function


Predicts the target values and prints R² score and MSE.



---

## Requirements

**Install the necessary Python libraries:**

pip install numpy pandas scikit-learn joblib matplotlib


---

## Usage

**Training**

*Run the following command to train the model and save the weights + scaler:*

python train.py

***This will:***

* Train the model on the dataset.

* Save the learned weights in MultiVariablelinearRegressionWeights.csv.npy.

* Save the fitted scaler in scaler.pkl.


**Testing**

After training, run:

python test.py

***This will:***

* Load the saved weights and scaler.

* Normalize the dataset.

* Compute predictions.

* Print R² score and MSE.



---

## Dataset

The default dataset used is Boston Housing (BostonHousing.csv).
However, you can replace it with any dataset where:

All columns except the last one are features.

The last column is the target variable.
