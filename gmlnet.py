# -*- coding: utf-8 -*-

'''
TODO : It is not perfect.
'''

__author__ = "Jeonghun Yoon"
import urllib2
import math
import matplotlib.pyplot as plot

def S(z, gamma):
    if gamma >= abs(z):
        return 0.0
    return (z/abs(z))*(abs(z) - gamma)

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
xy = [0.0] * n_col
for i in range(n_row):
    for j in range(n_col):
        xy[j] += x_data[i][j] * y_data[i]

max_xy = 0.0
for i in range(n_col):
    val = abs(xy[i])/n_row
    if val > max_xy:
        max_xy = val

# Start value of lambda
alpha = 1.0
lam = max_xy/alpha

beta = [0.0] * n_col

# Beta matrix initialize
beta_mat = []
beta_mat.append(list(beta))

# Setting for loop
n_steps = 100
lam_mult = 0.93 # recommanded by author

nz_list = []

for i_step in range(n_steps):
    lam = lam*lam_mult
    # Making lambda smaller
    delta_beta = 100.0
    epsilon = 0.01
    iter_step = 0
    beta_inner = list(beta)
    while delta_beta > epsilon:
        iter_step += 1
        if iter_step > 100:
            break
        beta_start = list(beta_inner)
        for i_col in range(n_col):
            xyj = 0.0
            for i in range(n_row):
                label_hat = sum([x_data[i][k]*beta_inner[k] for k in range(n_col)])
                residual = y_data[i] - label_hat
                xyj += x_data[i][i_col]*residual
            unc_beta = xyj/n_row + beta_inner[i_col]
            beta_inner[i_col] = S(unc_beta, lam*alpha) / (1+lam*(1-alpha))
        sum_diff = sum([abs(beta_inner[n] - beta_start[n]) for n in range(n_col)])
        sum_beta = sum([abs(beta_inner[n]) for n in range(n_col)])
        delta_beta = sum_diff / sum_beta
    print (i_step, iter_step)
    beta = beta_inner
    # Append new beta
    beta_mat.append(beta)
    # TODO : Why????
    nz_beta = [index for index in range(n_col) if beta[index] != 0.0]
    for q in nz_beta:
        if (q in nz_list) == False:
            nz_list.append(q)

# List of beta rank
name_list = [title[nz_list[i]] for i in range(len(nz_list))]
print (name_list)

n_pts = len(beta_mat)
for i in range(n_col):
    coef_curve = [beta_mat[k][i] for k in range(n_pts)]
    x_axis = range(n_pts)
    plot.plot(x_axis,coef_curve)
plot.xlabel("Steps Taken.")
plot.ylabel("Coefficient Values")
plot.show()