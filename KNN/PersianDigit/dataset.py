# import Libraries
import numpy as np
from scipy.io import loadmat
import cv2

# load dataset and Data preprocessing
def load_dataset(train_value : int = 1000,test_value : int = 200):
    
    # load dataset
    dataset = loadmat(r"C:\Users\pc\Documents\university file\Deep Learning\PersianDigit\Data_hoda_full.mat")
    
    # data spliting
    x_train = np.squeeze(dataset["Data"][:train_value])
    x_test = np.squeeze(dataset["Data"][train_value:train_value + test_value])
    y_train = np.squeeze(dataset["labels"][:train_value])
    y_test = np.squeeze(dataset["labels"][train_value:train_value + test_value])
    
    # resize
    x_train_10by10 = [ cv2.resize(img,(10,10)) for img in x_train]
    x_test_10by10 = [ cv2.resize(img,(10,10)) for img in x_test]

    # reshape
    x_train = np.reshape(x_train_10by10,(-1,100))
    x_test = np.reshape(x_test_10by10,(-1,100))

    return x_train,x_test,y_train,y_test
