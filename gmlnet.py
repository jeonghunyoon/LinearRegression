# -*- coding: utf-8 -*-

'''
1. data load
2. normalization
3. correlation coeifficeint
'''

__author__ = "Jeonghun Yoon"
import urllib2
import math
import numpy as np
import matplotlib.pyplot as plot

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
3. using correaltion coefficient update beta vector. started from a beta vector whose entryies are all 0.
'''
beta_vec = [0.0] * n_col
for i in range(n_row):
    for j in range(n_col):
        beta_vec[i][j] += x_data[i][j] * y_data[i]
