# -*- coding: utf-8 -*-
'''
    Coding K-nearest-neighbour from Scratch
    k = sqrt(n)
    Where n = number of Training Data
    Classify a random point
    Author: Tobias Stiemer
    Date: 22.11.2019
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import math

feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
data = pd.read_csv('iris.csv')
data = data.drop(columns=["Id"])
data = data.rename(columns={"species": "label"})

# Use Feature scaling to normalize Feature Values
# sc = StandardScaler()
# data[feature_names] = sc.fit_transform(data[feature_names])

# Picking a random dataset entry and using it as test
# 26    58  101
point = data.iloc[110]
data = data.drop(index=110)

# Picking an odd number for k
# If k is even, add 1 to make it uneven

k = math.sqrt(len(data["sepal_length"]))
k = int(k)
if (k % 2) == 0:
    k = k + 1

# Flatten out Data
data_f1, data_f2, data_f3, data_f4 = data["sepal_length"], data["sepal_width"], data["petal_length"], data[
    "petal_width"]

# Flatten out data to be able to use vectorization
point_f1, point_f2, point_f3, point_f4 = point.sepal_length, point.sepal_width, point.petal_length, point.petal_width

error_f1 = (data_f1 - point_f1) ** 2
error_f2 = (data_f2 - point_f2) ** 2
error_f3 = (data_f3 - point_f3) ** 2
error_f4 = (data_f4 - point_f4) ** 2

error_vector = error_f1 + error_f2 + error_f3 + error_f4
# print(error_vector)
data["error"] = error_vector
data = data.sort_values(by=['error'], ascending=True)
labels = data.iloc[:k, -2]
# print(type(labels))
df = labels.value_counts()
print(df)
