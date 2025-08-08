import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read dataset
df = pd.read_csv(r'C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part4\basic samples\sample-1\data\student_loan_train.csv')

# convert dataframe to an array
train = np.array(df)

# seprate data from label
train_x = train[:,0]
train_y = train[:,1]

# model
def linear_regression(w0,w1,x):
    y_hat = x * w1 + w0
    return y_hat

# MSE value - loss function
def mse(y,y_hat):
    loss = np.mean((y - y_hat)**2)
    return loss

# the derivative of the function
def grad_func(old_x):
    new_x = 0.2 * old_x
    return new_x

# gradient decent function
def gradient_decent(old_w, grad_func, eta, N):
    for _ in range(N):
        new_w = old_w - eta * grad_func(old_w)
        old_w = new_w
        print(old_w)
    return new_w


# optimizer function
def optimizer(x,y):
    w0 = 0
    w1 = np.linspace(1,100,10000)
    mse_list = []
    for i in w1:
        y_hat = linear_regression(w0,i,x)
        mse_value = mse(y,y_hat)
        mse_list.append(mse_value)
    loss_min_index = np.argmin(mse_list)
    loss_min = mse_list[loss_min_index]
    for i in zip(w1,mse_list):
        if i[1] == loss_min:
            w1_value = i[0]
            break
    return w1_value

best_w1 = optimizer(train_x,train_y)
y_hat = linear_regression(5,best_w1,train_x)

print(best_w1)

# show data points
plt.scatter(train_x,train_y)

# show linear regression model line
plt.plot(train_x,y_hat,'r')

plt.show()

