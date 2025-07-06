from sklearn.neighbors import KNeighborsClassifier
import cv2 as cv
from joblib import dump
from glob import glob

train_data = []
train_label = []

for address in glob(r"C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part3\Q1\train\*\*"):
    img = cv.imread(address)
    img = cv.resize(img,(32,32))
    img = img / 255.0
    img = img.flatten()
    train_data.append(img)

    x = address.split('\\')[-1].split('.')[0]
    train_label.append(x)
    

model = KNeighborsClassifier(51)
model.fit(train_data,train_label)

test_data = []
test_label = []

for address in glob(r"C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part3\Q1\test\*\*"):
    img = cv.imread(address)
    img = cv.resize(img,(32,32))
    img = img / 255.0
    img = img.flatten()
    test_data.append(img)

    x = address.split('\\')[-1].split('.')[0]
    test_label.append(x)

score = model.score(test_data,test_label)

print(f'your accurancy is = {score*100}%')

# dump(model,r'C:\Users\pc\Documents\university file\Machine Learning\Machine learning files\part3\Q1\cat-dog-model.joblib')
