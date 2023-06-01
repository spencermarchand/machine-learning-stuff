#Write a program which trains a neural network (MLP) to classify the data in the file ‘quad2.csv’. Use the first 200 samples for training and the rest for testing. Also, use a MLP with 3 layers, and 20 neurons per layer. 


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/spencermarchand/Documents/VS_code/Python/418/quad2.csv')
data = df.to_numpy()

x = data[:,0:2]
y = data[:,2]
X_train = x[0:200]
Y_train = y[0:200]
X_test = x[200:]
Y_test = y[200:]

model = MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter=1000)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print(accuracy_score(Y_test,y_pred))

xp1, xp2 = np.meshgrid(np.linspace(-2,2,1000),np.linspace(-2,2,1000))
# vectorize mesh grid 
xp1_v = xp1.reshape(-1,1) 
xp2_v = xp2.reshape(-1,1) 
# convert vectorized meshgrid to dataset 
Xp_data = np.append(xp1_v,xp2_v,axis=1) 
# transform features for plotting 
z = model.predict(Xp_data) 
z = z.reshape(xp1.shape) 
plt.contourf(xp1,xp2,z,alpha=.2) 
plt.show()