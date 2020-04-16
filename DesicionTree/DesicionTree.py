# -*- coding: utf-8 -*-
'''
    Coding a Decision Tree using sklearn
    Author: Tobias Stiemer
    Date: 13.11.2019
'''

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Loading Data - Initialize Dataframe
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
data = pd.read_csv('iris.csv')
data = data.drop(columns=["Id"])
data = data.rename(columns={"species": "label"})

feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = data[feature_columns]  # All my Features
y = data.label  # Labels of the Dataset

# Split dataset into training and validation (test) set - Size of test set = 20% of original dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# initialize Desicion Tree classifier object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Train the classifier
clf = clf.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Export as dot file
export_graphviz(clf, out_file='tree.dot',
                feature_names=feature_columns,
                class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                rounded=True, proportion=False,
                precision=2, filled=True)
