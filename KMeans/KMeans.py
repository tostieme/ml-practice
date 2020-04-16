# -*- coding: utf-8 -*-
'''
    Coding K-means from Scratch
    Data points are random
    k = 2
    Author: Tobias Stiemer
    Date: 22.11.2019
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

X = -2 * np.random.rand(100, 2)  # Numpy Array
X1 = 1 + 2 * np.random.rand(50, 2)
X[50:100, :] = X1
# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# initialize 2 random centroids
c1x = -0.5
c1y = -0.5
c2x = 2
c2y = 2
maxIter = 200
cluster1 = []
cluster2 = []

# calcuate the euclidian distance between the data points and the centroids
SubX = X[:, 0]
SubY = X[:, 1]
temp = 1

for x in range(maxIter):
    error_c1x = (X[:, 0] - c1x) ** 2
    error_c1y = (X[:, 1] - c1y) ** 2
    error_c1 = error_c1x + error_c1y

    error_c2x = (X[:, 0] - c2x) ** 2
    error_c2y = (X[:, 1] - c2y) ** 2
    error_c2 = error_c2x + error_c2y

    # True -> Cluster 1 | False -> Cluster 2
    boolArray_C = error_c1 < error_c2
    indicesC1 = np.where(boolArray_C == True)
    indicesC2 = np.where(boolArray_C == False)

    cluster1x = np.delete(SubX, indicesC2)
    cluster1y = np.delete(SubY, indicesC2)
    cluster2x = np.delete(SubX, indicesC1)
    cluster2y = np.delete(SubY, indicesC1)
    c1x = np.mean(cluster1x)
    c1y = np.mean(cluster1y)
    c2x = np.mean(cluster2x)
    c2y = np.mean(cluster2y)
    cluster1 = np.column_stack((cluster1x, cluster1y))
    cluster2 = np.column_stack((cluster2x, cluster2y))

print(c1x, c1y)
print(c2x, c2y)
plt.scatter(cluster1[:, 0], cluster1[:, 1], s=50, c="green")
plt.scatter(cluster2[:, 0], cluster2[:, 1], s=50, c="yellow")
plt.scatter(c1x, c1y, c='black', alpha=0.5)
plt.scatter(c2x, c2y, c='black', alpha=0.5)
plt.show()
