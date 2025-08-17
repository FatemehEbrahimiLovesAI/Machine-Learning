import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load weights
w = np.load(r'C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part5\F.Ebrahimi.boston\MultiVariablelinearRegressionWeights.csv.npy')

# Load dataset
df = pd.read_csv(r'C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part5\F.Ebrahimi.boston\BostonHousing.csv')

# Convert DataFrame to Numpy array
test = np.array(df)

# data spliting
test_x = test[:,:-1]
test_y = test[:,-1]

# Normalaize
scaler = load(r'C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part5\F.Ebrahimi.boston\scaler.pkl')
test_x = scaler.transform(test_x)

# Model
def linear_regression(x,w):
    y_hat = 0
    for xi,wi in zip(x.T,w):
        y_hat = y_hat + xi * wi
    return y_hat

# MSE function
def mse(y,y_hat):
    return np.mean((y_hat - y) ** 2)

# R2 score function
def R2(y,y_hat):
    return 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)

# Print the result
y_hat = linear_regression(test_x,w)
print(f'R2 score : {R2(test_y,y_hat)}\nMSE score : {mse(test_y,y_hat)}')


