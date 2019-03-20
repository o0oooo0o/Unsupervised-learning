# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 03:55:09 2019

@author: 321
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = np.genfromtxt('faithful.csv', delimiter=",")
pca = PCA(n_components=2).fit(data)

m = pca.mean_
v = pca.components_
#l = pca.explained_variance_

w = v[1,0]/v[0,0]
b = m[1]-w*m[0]

x=np.linspace(1.6,5.1, 1000)
y = w*x + b


cov = np.cov(data.T)
ev, eig = np.linalg.eig(cov)
mm = [np.mean(data[:,0]), np.mean(data[:,1])]

ww = eig[1,1]/eig[0,1]
bb = mm[1]-ww*mm[0]

yy = ww*x + bb

plt.subplots(1,2, figsize=(12,4))
plt.subplot(121)
plt.title("PCA sklearn")
plt.scatter(data[:,0], data[:,1], color='c', marker='.')
plt.scatter(m[0], m[1], color='b')
plt.plot(x,y, color='k')

plt.subplot(122)
plt.title("PCA numpy")
plt.scatter(data[:,0], data[:,1], color='c', marker='.')
plt.scatter(m[0], m[1], color='b')
plt.plot(x,yy, color='r')
plt.show()