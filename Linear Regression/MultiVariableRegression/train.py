import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import dump

# Load dataset
df = pd.read_csv(r'C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part5\F.Ebrahimi.boston\BostonHousing.csv')
train = np.array(df)

# Data split
train_x = train[:, :-1]
train_y = train[:, -1]

# Normalize and save the file for testS
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
dump(scaler,r'C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part5\F.Ebrahimi.boston\scaler.pkl')

# Add Bias term
m = train.shape[0]
bias = np.ones((m,1))
train_x = np.hstack((bias,train_x))
n = train_x.shape[1] 

# Initialize weights
w = np.random.randn(n)

# MSE function - loss function
def mse(y,y_hat):
    return np.mean((y - y_hat ) ** 2)

# Model
def linear_regression(x,w):
    y_hat = []

    for row in x:
        pre = 0 
        for xi,wi in zip(row,w):
            pre = pre + ( xi * wi ) 
        y_hat.append(pre)
    return y_hat

# Gradient computation
def gradients(x,y,y_hat):
    grads = []
    for xi in x.T:
        grads.append(2 * np.mean(xi * (y_hat - y)))
    return np.array(grads) 

# Gradient decent
def gradient_decent(w,eta,grads):
    return w - (eta * grads)

# Just some stars
stars = '*' * 30

# Main code
erorr_list = []
for epoch in range(100):
    y_hat = linear_regression(train_x,w)
    e = mse(train_y,y_hat)
    erorr_list.append(e)
    grads = gradients(train_x,train_y,y_hat)
    w = gradient_decent(w,0.1,grads)
    print(f'the epoch : {epoch}\nits error : {e}\nits weights : {w}\n{stars}\n')

# Save the weights for test files
np.save(r'C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part5\F.Ebrahimi.boston\MultiVariablelinearRegressionWeights.csv',w)
