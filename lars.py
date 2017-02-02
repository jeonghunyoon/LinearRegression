# -*- coding: utf-8 -*-

'''
1. data load
2. training set/ test set
3. normalization
4. correlation check
5. beta update
'''

__author__ = "Jeonghun Yoon"
import urllib2
import math
import numpy as np
import matplotlib.pyplot as plot
from sklearn.cross_validation import train_test_split

'''
1. data load from uci archieve
'''
data = urllib2.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
x_data = []
y_data = []
title = []
is_first = True

for row in data.readlines():
    if is_first is True:
        title = row.strip().split(";")
        is_first = False
    else:
        tokens = row.strip().split(";")
        # label for feature vector
        y_data.append(float(tokens[-1]))
        del(tokens[-1])
        # feature vector
        x_data.append(map(float, tokens))


'''
2. splitting to training set and test set
'''
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)


'''
3. normalization for trainin set
'''
n_row = len(x_train)
n_col = len(x_train[0])

# normalization of feature vector
for j, _ in enumerate(x_train[0]):
    # mean
    mean = sum([x_train[i][j] for i, _ in enumerate(x_train)]) / float(n_row)
    # variance
    variance = sum([(x_train[i][j] - mean) ** 2 for i, _ in enumerate(x_train)]) / float(n_row)
    # standard deviation
    stddev = math.sqrt(variance)
    # normalization
    for i, _ in enumerate(x_train):
        x_train[i][j] = (x_train[i][j] - mean) / stddev

# normalization of label
# mean
mean = sum(y_train) / float(n_row)
# variance
variance = sum([(v - mean) ** 2 for _, v in enumerate(y_train)]) / float(n_row)
# standard deviation
stddev = math.sqrt(variance)
# normalization
for i, v in enumerate(y_train):
    y_train[i] = (v - mean) / stddev


'''
4. correlation
5. beta updates
'''
# initiate beta vectors
beta = [0.0] * n_col
beta_matrix = []
beta_matrix.append(beta)

# TODO: How to get number of step, step size?
n_step = 350
step_size = 0.004

# use numpy matrix
x_train_mat = np.matrix(x_train)

# residual
for stpe in range(n_step):
    residuals = []
    for i, v in enumerate(x_train):
        y_hat = np.dot(np.array(v), np.array(beta))
        residuals.append(y_train[i] - y_hat)
    corrs = []
    for j in range(n_col):
        # TODO why? correlation? how to derive?
        corr = np.dot(x_train_mat[:, j].T, np.matrix(residuals).T) / n_row
        corrs.append(float(corr))
    # find maximum of abs(residual)
    i_max = 0
    res_ab_max = 0
    for i, v in enumerate(corrs):
        if abs(res_ab_max) < abs(v):
            i_max = i
            res_ab_max = v
    # TODO: How to gaurante the size of step_size?
    beta[i_max] += step_size * res_ab_max / abs(res_ab_max)
    beta_matrix.append(beta[:])

'''
visualization
'''
for i in range(n_col):
    coef_curve = [beta_matrix[k][i] for k in range(n_step)]
    x_axis = range(n_step)
    plot.plot(x_axis, coef_curve)

plot.xlabel("Step Taken")
plot.ylabel("Coefficient Values")
plot.show()