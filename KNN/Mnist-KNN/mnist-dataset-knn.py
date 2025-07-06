import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


train = pd.read_csv(r'data\mnist_train.csv',header=None)
test = pd.read_csv(r'data\mnist_test.csv',header=None)

train_data = train.iloc[:,1:]
train_label = train.iloc[:,0]

test_data = test.iloc[:,1:]
test_label = test.iloc[:,0]

model = KNeighborsClassifier(3)
model.fit(train_data,train_label)

accurancy = model.score(test_data,test_label)

print(f"This is the model's accurancy : {accurancy*100}%")
