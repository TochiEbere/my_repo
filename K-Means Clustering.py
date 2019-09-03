# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:02:33 2019

@author: TOCHI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')

X = df.iloc[:, [3,4]].values

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()


#Applying the kmeans model to the mall dataset
km = KMeans(n_clusters=5, random_state=0)
cluster = km.fit_predict(X)

#visulizing the clusters
plt.scatter(X[cluster==0, 0], X[cluster==0,1], c='red')
plt.scatter(X[cluster==1, 0], X[cluster==1,1], c='yellow')
plt.scatter(X[cluster==2, 0], X[cluster==2,1], c='green')
plt.scatter(X[cluster==3, 0], X[cluster==3,1], c='magenta')
plt.scatter(X[cluster==4, 0], X[cluster==4,1], c='cyan')
plt.show()




