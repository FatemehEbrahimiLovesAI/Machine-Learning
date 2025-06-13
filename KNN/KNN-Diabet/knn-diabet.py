import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
	
# import dataset
df = pd.read_csv(r'C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part2\diabetes.csv')
 
# Separating data from the target
data = df.iloc[:,0:-1]
label = df.iloc[:,-1]

# Removing zero data and replacing it with the middle number of the column 
no_zero =["Glucose", "BloodPressure","SkinThickness","Insulin","BMI"]

for col in no_zero:
    df[col] = df[col].replace(0,np.nan)
    mean = int(df[col].mean(skipna=True))
    df[col] = df[col].replace(np.nan,mean)

# Separating test data from train data
x_train, x_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=12)

# Scaling datas
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) 

# creating the model
model = KNeighborsClassifier(33)
model.fit(x_train,y_train)

# Predicting test data labels
pre = model.predict(x_test)

# Measuring model accurancy 
acc = accuracy_score(y_test,pre)

print(acc)
