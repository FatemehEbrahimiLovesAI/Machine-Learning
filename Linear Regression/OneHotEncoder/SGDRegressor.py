import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

# Import dataset with column names
cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
df = pd.read_csv(
    'HousesInfo.csv',
    sep=" ",
    header=None,
    names=cols
)

# Remove rare zipcodes (those with fewer than 25 occurrences)
zipcodes = df['zipcode'].value_counts().keys().to_list()
counts = df["zipcode"].value_counts().to_list()
for (zipcode, count) in zip(zipcodes, counts):
    if count < 25:
        index = df[df["zipcode"] == zipcode].index
        df.drop(index, inplace=True)

# Split dataset into training (80%) and testing (20%)
train, test = train_test_split(df, test_size=0.2, random_state=20)

# Scale continuous features (bedrooms, bathrooms, area)
colno = ["bedrooms", "bathrooms", "area"]
scaler = StandardScaler()
train_normal_con = scaler.fit_transform(train[colno])
test_normal_con = scaler.transform(test[colno])

# Encode categorical feature (zipcode) into one-hot vectors
one = LabelBinarizer()
train_normal_cat = one.fit_transform(train["zipcode"])
test_normal_cat = one.transform(test["zipcode"])

# Combine continuous and categorical features
train_x = np.hstack((train_normal_con, train_normal_cat))
test_x = np.hstack((test_normal_con, test_normal_cat))

# Normalize target variable (price) by dividing by max price
max_price = df['price'].max()
train_y = train["price"] / max_price
test_y = test["price"] / max_price

# Initialize and train the regression model using SGD
model = SGDRegressor(eta0=0.1, max_iter=100)
model.fit(train_x, train_y)

# Make predictions on the test set
pre = model.predict(test_x)

# Evaluate model performance using MSE and R² score
mse = mean_squared_error(test_y, pre)
r2 = r2_score(test_y, pre)

# Print evaluation results
print(f'your mse score : {mse}\nyour R2 score : {r2}')
