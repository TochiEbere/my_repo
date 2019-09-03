# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:54:01 2019

@author: TOCHI
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection as ms, neighbors

df = pd.read_csv('breast-cancer-wisconsin.data')
df.drop('id', 1, inplace=True)

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])


x_train, x_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=101)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test) 
