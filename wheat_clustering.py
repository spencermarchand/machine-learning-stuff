from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

# Load the data
df = pd.read_csv('/Users/spencermarchand/Documents/VS code/Python/418/seeds.csv')
data = df.to_numpy()
X = data[:,0:6]
y = data[:,7]

U = X - np.mean(X,axis=0)
C = PCA(2)
C.fit(U)
x1_pca = C.transform(U)

# Plot the data
plt.scatter(x1_pca[y==0,0],x1_pca[y==0,1])
plt.scatter(x1_pca[y==1,0],x1_pca[y==1,1])
plt.scatter(x1_pca[y==2,0],x1_pca[y==2,1])
plt.show()

#generate scree plot

K_Max =10
sum_squared_distance = []
for k in range(1,K_Max):
    kmeans = KMeans(k)
    kmeans.fit(x1_pca)
    sum_squared_distance.append(kmeans.inertia_)

plt.plot(range(1,K_Max), sum_squared_distance)
plt.xlabel('k')
plt.ylabel("sum of squared distance")
plt.show()

k = 3
kmeans = KMeans(k)
kmeans.fit(x1_pca)
a = kmeans.predict(x1_pca)
centroid = kmeans.cluster_centers_
plt.scatter(x1_pca[a==0,0],x1_pca[a==0,1])
plt.scatter(x1_pca[a==1,0],x1_pca[a==1,1])
plt.scatter(x1_pca[a==2,0],x1_pca[a==2,1])
plt.show()


