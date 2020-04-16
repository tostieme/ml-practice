# -*- coding: utf-8 -*-
'''
    Coding a Random Forest using sklearn
    Author: Tobias Stiemer
    Date: 13.11.2019
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataset = pd.read_csv("iris.csv")
dataset = dataset.drop(columns=["Id"])
dataset = dataset.rename(columns={"species": "label"})

feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = dataset[feature_columns]  # All my Features
y = dataset.label  # Labels of the Dataset
# Divide into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = RandomForestClassifier(n_estimators=20, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))