# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:13:28 2019

@author: 321
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

k=2
data = np.genfromtxt('faithful.csv', delimiter=",")
kmeans = KMeans(n_clusters=k).fit(data)

predict = kmeans.predict(data)

centers = np.hstack([data[np.random.randint(data.shape[0], size=(k,1)),0],
                     np.random.randint(np.min(data[:,1]), np.max(data[:,1]), size=(k,1))])
def distances(centers, data):
    return np.linalg.norm(data-centers, axis=1)

rot = 10
labels = np.zeros(data.shape[0])
for num in range(rot):
    for i in range(data.shape[0]):
        
        dis=distances(centers, data[i])
        labels[i] =  np.argmin(dis)
        
    for k in range(k):
        centers[k] = np.mean(data[labels==k], axis=0)


plt.subplots(1,2, figsize=(12,4))
plt.subplot(121)
plt.title("K-means numpy(rot={})".format(rot))
plt.scatter(data[labels==0,0], data[labels==0,1], color='m', marker='.')
plt.scatter(data[labels==1,0], data[labels==1,1], color='c', marker='.')
plt.scatter(centers[:,0], centers[:,1], color='k', marker='+')

plt.subplot(122)
plt.title("K-means sklearn")
plt.scatter(data[predict==0,0], data[predict==0,1], color='c', marker='.')
plt.scatter(data[predict==1,0], data[predict==1,1], color='m', marker='.')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='b', marker='+')
plt.show()