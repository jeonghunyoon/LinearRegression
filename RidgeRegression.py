# -*- coding: utf-8 -*-

'''
<Description>
모델의 과적합을 피하기 위한 방법에는,
1. 최량 부분 집합 회귀
2. 단계적 전진 회귀
3. 계수 페널라이즈드 회귀 (Coefficient penalized regression) 이 있다.

1, 2 번의 방법은 계수 중 일부를 0으로 만든다.
3 번의 방법은 모든 계수를 더 작게 만든다.

여기서는 3 번의 방법 중, Ridge regression 을 다룰 것이다.

<Detail of implementation>
(1) training set과 test set의 비율은 전체 데이터를 기준으로 7:3의 비율로 나눌 것이다.
(2) training 은, SK-learn의 Ridge linear regression을 사용할 것이다.
(3) training 후, 모델의 성능을 측정하기 위하여 RMSE를 사용할 것이다. (Root mean squared error)
'''

from urllib2 import urlopen
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


##############################################################
# (1) 데이터를 load 한다. training set과 test set의 비율은 7:3 이다.
##############################################################
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
data = urlopen(url)
x_data = []
y_data = []
title = []
first_line = True

for line in data.readlines():
    if first_line:
        title = line.strip().split(";")
        first_line = False
    else:
        tokens = line.strip().split(";")
        # label 데이터
        y_data.append(float(tokens[-1]))
        del (tokens[-1])
        # feature vector
        x_data.append(map(float, tokens))

# training data : test data = 7:3
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)


############################
# (2) model training 을 한다.
############################
# ridge regression 에서 penalty 의 scale 을 조정하는 coefficient. 값이 작을수록 ordinary 선형 회귀와 동일해진다.
alpha_list = [0.1**i for i in [0, 1, 2, 3, 4, 5, 6]]
model_list = []

for alpha in alpha_list:
    ridge_model = linear_model.Ridge(alpha=alpha)
    ridge_model.fit(x_train, y_train)
    model_list.append(ridge_model)


##########################
# (3) model의 성능을 측정한다.
##########################
# 모델의 성능을 측정하기 위해서, RMSE를 사용한다.
rmse_list = []

for model in model_list:
    # RMSE를 계산한다.
    error_vector = model.predict(x_test) - y_test
    mse = sum([error**2 for error in error_vector]) / len(error_vector)
    rmse_list.append(np.sqrt(mse))

# 결과를 출력
for i in range(len(rmse_list)):
    print("alpha : ", alpha_list[i], ", rmse : ", rmse_list[i])


#########################################
# Ordinary Linear Regression의 rmse와 비교
#########################################
# linear model training
lin_model = linear_model.LinearRegression()
lin_model.fit(x_train, y_train)

# RMSE 측정
lin_error = lin_model.predict(x_test) - y_test
mse = sum([error**2 for error in lin_error]) / len(lin_error)
rmse = np.sqrt(mse)

print ("linear regression : ", rmse)


###############
# (4) Plotting
###############
x_axis = range(len(rmse_list))
plt.figure()
plt.plot(x_axis, rmse_list, "k")
plt.xlabel("-log(alpha)")
plt.ylabel("Error (RMSE)")
plt.show()