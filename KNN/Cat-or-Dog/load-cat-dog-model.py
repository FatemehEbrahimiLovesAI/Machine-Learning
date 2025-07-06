import cv2 as cv
from glob import glob
from joblib import load
import numpy as np

model = load('cat-dog-model.joblib')

data = []
label = []

for address in glob('your_dataset_folder\*'):

    img = cv.imread(address)
    img_r = cv.resize(img,(32,32))
    img_r = img_r / 255.0
    img_r = img_r.flatten()
    data.append(img_r)

    img_r = np.array([img_r])
    pre = model.predict(img_r)[0]

    x = address.split('\\')[-1].split('.')[0]
    label.append(x)

    cv.putText(img,str(pre),(250,250),cv.FONT_HERSHEY_SCRIPT_COMPLEX,2,(230,216,173),5)
    cv.imshow("image",img)
    cv.waitKey(0)

print(f'the accurancy score is :{model.score(data,label)*100}%')
