# House Price Prediction with SGDRegressor

## Overview

This project demonstrates a machine learning pipeline for predicting house prices using Stochastic Gradient Descent Regressor (SGDRegressor) from scikit-learn.

The dataset includes house features such as number of bedrooms, bathrooms, area, zipcode, and price. Both numerical and categorical features are processed and used for model training.


---

## Features

Data Cleaning: Removes rare zipcodes with fewer than 25 samples.

**Preprocessing:**

* Scales numerical features (bedrooms, bathrooms, area) using StandardScaler.

* Encodes categorical feature (zipcode) using one-hot encoding (LabelBinarizer).


Target Normalization: Normalizes house prices by dividing by the maximum price.

***Model***: Trains an SGDRegressor with a learning rate (eta0) of 0.1 and 100 iterations.

**Evaluation**:

* Mean Squared Error (MSE)

* R² Score




---

## Requirements

* Python 3.x

* pandas

* numpy

* scikit-learn


**Install dependencies:**

pip install pandas numpy scikit-learn


---

## Usage

**Run the script:**

python SGDRegressor.py


**Example output:**

your mse score : 0.025
your R2 score : 0.87




---

## Notes

Rare zipcodes (with <25 houses) are dropped to prevent overfitting.

Prices are normalized to ensure stable model training.

The model can be tuned by adjusting parameters like eta0 and max_iter.
