import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,zero_one_loss

# Importing a dataset and creating a dataframe
df = pd.read_csv("Iris.data",header= None)

# Separating labels from features
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]

# Separating test and train values ​​and placing them inside variables
x_tran, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# Creating a KNN model
clf = KNeighborsClassifier(5)

# Training the model
clf.fit(x_tran,y_train)

# Testing the model
pre = clf.predict(x_test)

# Estimate how well the model is working.
acc = accuracy_score(y_test,pre)
loss = zero_one_loss(y_test,pre)

# Printing the result
print(f"Correct guesses : {acc*100}\nWrong guesses : {loss*100}")
