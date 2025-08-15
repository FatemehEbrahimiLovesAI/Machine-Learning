### Linear Regression from Scratch – Boston Housing Dataset

## Overview

This project implements Linear Regression from scratch (without using sklearn's regression model) to predict housing prices from the Boston Housing dataset.
It uses gradient descent for parameter optimization and Mean Squared Error (MSE) as the loss function.
The code is designed to be modular and easy to extend — in the future, a test section will be added to evaluate the model on unseen data.


---

## Features

Data loading from CSV file using pandas

Feature scaling with StandardScaler

Adding a bias term manually

Custom implementation of:

Linear Regression model

Mean Squared Error (MSE) loss function

Gradient computation

Gradient Descent optimization


Epoch-based training with detailed logging of:

Epoch number

Current error (MSE)

Updated weights




---

## Requirements

Make sure you have Python installed along with the required libraries:

pip install numpy pandas scikit-learn


---

## How to Run

1. Download the Boston Housing dataset as BostonHousing.csv and place it in your specified path.


2. Update the file path in the code:

df = pd.read_csv(r'C:\path\to\BostonHousing.csv')


3. Run the Python script:

python linear_regression.py




---

## Code Structure

Data Preparation: Load and scale the dataset, add bias term.

Model Functions:

mse(y, y_hat): Calculates the Mean Squared Error.

linear_regression(x, w): Predicts target values given features and weights.

gradients(x, y, y_hat): Computes the gradients for each weight.

gradient_decent(w, eta, grads): Updates weights using gradient descent.


Training Loop: Runs for a fixed number of epochs, printing performance metrics after each epoch.

---

## Future Improvements

Add a test phase to evaluate performance on unseen data.

its error : 81.234
its weights : [0.12, -0.34, 0.45, ...]
**************************
