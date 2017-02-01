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