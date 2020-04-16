# -*- coding: utf-8 -*-
'''
    Coding Linear Regression from Scratch
    Author: Tobias Stiemer
    Date: 12.11.2019
    Linear Regression fitted with Least Squares
    y = mx+b where
    m   = sum((x(i)-mean(x))*sum((y(i)-mean(y))
        = covariance(x,y)/variance(x)
    b   = (sum(y) - m*sum(x))/n     - n-training samples

    covariance and variance will be calculated without dividing by n-1,
    since it will be canceled (both of the terms denominators have n-1 in them)
'''

from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Loading Training-Set - Initialize Dataframe ---#
training_data = pd.read_csv('train.csv')

# --- Loading Test-Set - Initialize Dataframe ---#
validation_data = pd.read_csv('test.csv')

# --- Splitting the Rows to calculate the mean, covariance, variance ---#
X_train = training_data['x'].to_numpy()
Y_train = training_data['y'].to_numpy()
# print("Mean x = "+str(np.mean(X_train))


# --- Splitting the Rows for Val_Set ---#
X_test = validation_data['x'].to_numpy()
Y_test = validation_data['y'].to_numpy()


# --- calculating the covariance of x,y without dividing by n-1 ---#
def covariance(X_train, Y_train):
    # x(i)-mean(x)
    X_train = X_train - np.mean(X_train)

    # y(i)-mean(y)
    Y_train = Y_train - np.mean(Y_train)

    # x(i)-mean(x) * y(i)-mean(y)
    factor = X_train * Y_train

    # sum(x(i)-mean(x) * y(i)-mean(y))
    covariance_xy = np.sum(factor)
    return covariance_xy


# --- calculating the variance of x without dividing by n-1 ---#
def variance(X_train):
    # (x(i) - mean(x))²
    variance_x = np.sum((X_train - np.mean(X_train)) ** 2)
    return variance_x


# --- calculating the coefficients for the linear regression model y = mx + b ---#
def coefficients(X_train, Y_train):
    # m = covariance(x,y)/variance(x)    -> Slope
    # B = mean(y) - m*mean(x)            -> Intercept
    m = covariance(X_train, Y_train) / variance(X_train)
    b = np.mean(Y_train) - m * np.mean(X_train)
    return [m, b]


# --- Performing the linear Regression ---#
def linear_regression(X_train, Y_train, x_ValSet):
    preds = list()
    m, b = coefficients(X_train, Y_train)
    for x in x_ValSet:
        y_pred = m * x + b
        preds.append(y_pred)
    return preds


# --- Calculate root mean squared error ---#
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


# --- Calculatiing RMSE - closer to 0 means better fitted model ---#
def get_rmse(Y_test, predictions):
    return np.sqrt((np.sum((predictions - Y_test) ** 2)) / len(Y_test))


# --- Calculating R² Score - Ranges from 0 to 1, with 1 being perfect correlation ---#
def r_sqaured_score(X_test, X_train, Y_train, Y_test, predictions):
    sumofsquares = 0
    sumofresiduals = 0
    m, b = coefficients(X_train, Y_train)
    for x in X_test:
        yhat = m * x + b
        sumofsquares += np.sum((predictions - np.mean(Y_test)) ** 2)
        sumofresiduals += np.sum((predictions - Y_test) ** 2)
    score = 1 - (sumofresiduals / sumofsquares)
    return score


# --- Plotting the Data and the fitted Line ---#
def createPlot(X_train, X_test, Y_train):
    x_max = np.max(X_train) + 100
    x_min = np.min(X_test) - 100
    # calculating line values of x and y
    x = np.linspace(x_min, x_max, 1000)
    m, b = coefficients(X_train, Y_train)
    y = b + m * x
    # plotting line
    plt.plot(x, y, color='#00ff00', label='Linear Regression')
    # plot the data point
    plt.scatter(X_train, Y_train, color='#ff0000', label='Data Point')
    # x-axis label
    plt.xlabel('Head Size (cm^3)')
    # y-axis label
    plt.ylabel('Brain Weight (grams)')
    plt.legend()
    plt.show()


# createPlot(X_train,X_test,Y_train)
predictions = linear_regression(X_train, Y_train, X_test)
print(get_rmse(Y_test, predictions))
print(r_sqaured_score(X_test, X_train, Y_train, Y_test, predictions))
