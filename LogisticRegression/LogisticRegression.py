# -*- coding: utf-8 -*-
'''
    Coding Logistic Regression
    Author: Tobias Stiemer
    Date: 22.11.2019
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns


sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)
data = pd.read_csv('iris.csv')
data = data.rename(columns={"species": "label"})
data = data.drop(columns=["Id"])
X = data.iloc[:, 0:4]
y = data.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(score)
