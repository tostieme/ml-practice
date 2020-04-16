# -*- coding: utf-8 -*-
'''
    Coding Linear Regression from Scratch
    Author: Tobias Stiemer
    Date: 13.11.2019
    Linear Regression fitted with Gradient Descent

    y = mx +b
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Loading Training-Set - Initialize Dataframe ---#
training_data = pd.read_csv('train.csv')
X_train = training_data.iloc[:, 0]
Y_train = training_data.iloc[:, 1]

# --- Loading Test-Set - Initialize Dataframe ---#
validation_data = pd.read_csv('test.csv')
X_test = validation_data.iloc[:, 0]
Y_test = validation_data.iloc[:, 1]

m = 0
b = 0
learningRate = 0.0001
epochs = 1000
n = float(len(X_train))


# --- Gradient Descent for optimal Coefficients ---#
def gradientdescent(X_train, Y_train, learningRate, m, b, n):
    for i in range(epochs):
        Y_pred = m * X_train + b
        D_m = (-2 / n) * sum(X_train * (Y_train - Y_pred))
        D_b = (-2 / n) * sum(Y_train - Y_pred)
        m = m - learningRate * D_m
        b = b - learningRate * D_b
    return [m, b]


# --- Performing the linear Regression ---#
def linear_regression(X_train, Y_train, X_test, learningRate, m, b, n):
    preds = list()
    m, b = gradientdescent(X_train, Y_train, learningRate, m, b, n)
    for x in X_test:
        y_pred = m * x + b
        preds.append(y_pred)
    return preds


# --- Calculatiing RMSE - closer to 0 means better fitted model ---#
def get_rmse(Y_test, predictions):
    return np.sqrt((np.sum((predictions - Y_test) ** 2)) / len(Y_test))


# --- Calculating R² Score - Ranges from 0 to 1, with 1 being perfect correlation ---#
def r_sqaured_score(X_test, X_train, Y_train, Y_test, predictions, learningRate, m, b, n):
    sumofsquares = 0
    sumofresiduals = 0
    m, b = gradientdescent(X_train, Y_train, learningRate, m, b, n)
    for x in X_test:
        yhat = m * x + b
        sumofsquares += np.sum((predictions - np.mean(Y_test)) ** 2)
        sumofresiduals += np.sum((predictions - Y_test) ** 2)
    score = 1 - (sumofresiduals / sumofsquares)
    return score


# --- Plotting the Data and the fitted Line ---#
def createPlot(X_test, X_train, Y_train, m, b, n, learningRate):
    m, b = gradientdescent(X_train, Y_train, learningRate, m, b, n)
    print(m, b)
    Y_pred = m * X_test + b
    x_max = np.max(X_train) + 100
    x_min = np.min(X_test) - 100
    # calculating line values of x and y
    x = np.linspace(x_min, x_max, 1000)
    # plotting line
    plt.plot(X_test, Y_pred, color='#00ff00', label='Linear Regression')
    # plot the data point
    # plt.scatter(X_train,Y_train)
    plt.scatter(X_train, Y_train, color='#ff0000', label='Data Point')
    # x-axis label
    plt.xlabel('X')
    # y-axis label
    plt.ylabel('Y')
    plt.legend()
    plt.show()


predictions = linear_regression(X_train, Y_train, X_test, learningRate, m, b, n)
print("RMSE: " + str(get_rmse(Y_test, predictions)))
print("R²: " + str(r_sqaured_score(X_test, X_train, Y_train, Y_test, predictions, learningRate, m, b, n)))
createPlot(X_test, X_train, Y_train, m, b, n, learningRate)
