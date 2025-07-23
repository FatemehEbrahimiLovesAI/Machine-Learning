import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score

# Read dataset
df = pd.read_csv(r'C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part4\student score\student_scores.csv')

# Separating label from data
data = df[['Hours']]
label = df['Scores']

# Separating train and test data
data_train , data_test, label_train , label_test = train_test_split(data,label,test_size=0.2,random_state=12)

# Creating a model 
model = LinearRegression()

# Training the model
model.fit(data_train,label_train)

# Predicting test data labels
predictions = model.predict(data_test)

# Model evaluation
mae = mean_absolute_error(label_test,predictions)
mse = mean_squared_error(label_test,predictions)

# Printing model evaluation results
print(f'MAE score:{mae}\nMSE score:{mse}')

# Printing the results visually
plt.figure(figsize=(10,5))  
plt.scatter(data,label,color='lightblue',edgecolors='k',alpha=0.7)
plt.title('Hours vs Percentage')
plt.xlabel('Hours')
plt.ylabel('Percentage Score')
a = model.coef_
b = model.intercept_
y = a*data+b
plt.plot(data,y,'lightblue')
plt.show()
