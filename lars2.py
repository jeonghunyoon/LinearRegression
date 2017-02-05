# -*- coding: utf-8 -*-

'''
1. data load
2. normalization
3. 10 cross-validation
 3-1. correlation check
 3-2. beta update
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
2. normalization for trainin set
'''
n_row = len(x_data)
n_col = len(x_data[0])

# normalization of feature vector
for j, _ in enumerate(x_data[0]):
    # mean
    mean = sum([x_data[i][j] for i, _ in enumerate(x_data)]) / float(n_row)
    # variance
    variance = sum([(x_data[i][j] - mean) ** 2 for i, _ in enumerate(x_data)]) / float(n_row)
    # standard deviation
    stddev = math.sqrt(variance)
    # normalization
    for i, _ in enumerate(x_data):
        x_data[i][j] = (x_data[i][j] - mean) / stddev

# normalization of label
# mean
mean = sum(y_data) / float(n_row)
# variance
variance = sum([(v - mean) ** 2 for _, v in enumerate(y_data)]) / float(n_row)
# standard deviation
stddev = math.sqrt(variance)
# normalization
for i, v in enumerate(y_data):
    y_data[i] = (v - mean) / stddev


'''
3. 10 cross validation
 3-1. correlation check
 3-2. beta update
'''
# 10 cross validation
n_cross_val = 10

# TODO: How to get number of step, step size?
n_step = 350
step_size = 0.004

# total average of sum of squares of residuals
errors = []
for _, _ in enumerate(n_step):
    errors.append([])

# 10 fold
for i_trial in range(n_cross_val):
    # initiate beta vectors
    beta = [0.0] * n_col
    beta_matrix = []
    beta_matrix.append(beta)

    # index for training data, test data
    idx_test = [i for i in range(n_row) if i%n_cross_val == i_trial]
    idx_train = [i for i in range(n_row) if i % n_cross_val != i_trial]

    # training data, test data
    x_train = [x_data[i] for i in idx_train]
    y_train = [y_data[i] for i in idx_train]
    x_test = [x_data[i] for i in idx_test]
    y_test = [y_data[i] for i in idx_test]

    # use numpy matrix
    x_train_mat = np.matrix(x_train)
    x_test_mat = np.matrix(x_test)
    y_test_mat = np.matrix(y_test)

    # residual
    for step in range(n_step):
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

        # errors for mse
        for i, v in enumerate(y_test):
            error = v - np.dot(x_test_mat[i, :], np.array(beta))
            errors[step].append(error)


'''
4. mse
'''
mse_list = []
for err_vec in errors:
    mse = sum([error * error for error in err_vec]) / len(err_vec)
    # len(mse_list) = 350
    mse_list.append(mse)

idx_min_mse = 0
min_mse = 0
for i, v in mse_list:
    if v < min_mse:
        idx_min_mse = i
        min_mse = v

print "Minimum MSE : %f" %(min_mse)
print "Minimum MSE index : %d" %(idx_min_mse)

'''
visualization
'''
x_axis = range(len(mse_list))
plot.plot(x_axis, mse_list)

plot.xlabel("Step Taken")
plot.ylabel("MSE")
plot.show()