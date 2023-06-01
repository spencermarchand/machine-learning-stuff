import os
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image


folder_training = ['/Users/spencermarchand/Documents/VS code/Python/MNIST/training/' + str(i) + '/' for i in range(10)]
folder_testing = ['/Users/spencermarchand/Documents/VS code/Python/MNIST/testing/' + str(i) + '/' for i in range(10)]
classes = [i for i in range(10)]
def get_data(folder,im_width,label,n_samples): 
    file_names = os.listdir(folder)
    x = np.empty((n_samples,im_width**2))
    y = np.empty((n_samples,1))
    for i in range(n_samples):
        path = folder +file_names[i]
        im = Image.open(path).convert('L')
        im = im.resize((im_width,im_width))
        im_array = asarray(im)
        x[i,:] = im_array.reshape(1,-1)
        y[i,:] = classes[label]
        return x,y

im_width = 8
P_per_class = 1000
x_train = np.empty((P_per_class*10,im_width**2))
y_train = np.empty((P_per_class,1))

for i in range(10):
    x_i, y_i = get_data(folder_training[i],im_width,i,P_per_class)
    x_train[i*P_per_class:(i+1)*P_per_class,:] = x_i
    y_train[i*P_per_class:(i+1)*P_per_class,:] = y_i

print(x_train.shape, y_train.shape)
model = LogisticRegression()
model.fit(x_train, y_train)
P_per_class = 800
x_test = np.empty((P_per_class*10,im_width**2))
y_test = np.empty((P_per_class*10,1))

for i in range(10):
    x_i,y_i = get_data(folder_testing[i],im_width,P_per_class)
    x_train[i*P_per_class:(i+1)*P_per_class,:] = x_i
    y_train[i*P_per_class:(i+1)*P_per_class,:] = y_i

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


    





