#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, .1)

print("y is:")
print(y_train)

def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    def K(x_n, x_star):
        return np.exp(-np.power(np.linalg.norm(x_n - x_star), 2) / tau)
    y_test = np.zeros(len(x_test))
    for i, x_star in enumerate(x_test):
        distances = [(K(x_n, x_star), y_n) for x_n, y_n in data]
        # break ties in favor of higher x values for consistency with staff tests
        nearest = sorted(distances, key = lambda t: t[0])
        f_x_star = sum([y_n for _, y_n in nearest[-k:]]) / k
        y_test[i] = f_x_star
    return y_test

def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0,3])
    
    y_test = predict_knn(k=k)
    
    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.legend()
    plt.title("KNN Predictions with k = " + str(k))
    plt.savefig('k' + str(k) + '.png')
    plt.show()

for k in (1, 3, len(x_train)-1):
    plot_knn_preds(k)