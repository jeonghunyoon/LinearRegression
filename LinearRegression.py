# -*- coding: utf-8 -*-
# Linear Regression을 공부하는 코드를 작성하도록 한다. 코드는 재사용성 있게 짜도록 한다.
# data는 uci repository에 있는 wine data를 사용할 것이다.
# 데이터를 먼저 보고 코드를 짜도록 한다. (가장 중요함)

__author__ = 'Jeonghun Yoon'
import urllib
import random
import numpy as np

from sklearn import linear_model

### 1. Load data from uci repository.
xData = []
yData = []

f = urllib.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')
lines = f.readlines()
# Extract the title.
titles = lines.pop(0)

for line in lines:
    tokens = line.strip().split(';')
    # target values를 따로 yData에 저장한다.
    yData.append(float(tokens[-1]))
    del(tokens[-1])
    # feature values를 xData에 저장한다.
    xData.append(map(float, tokens))


### 2. Divide data set into two parts, train set(70%) and test set(30%).
nTest = int(len(xData) * 0.3)
# testIdx를 구한다.
testIdx = random.sample(range(len(xData)), nTest)
testIdx.sort()
# trainIdx를 구한다.
trainIdx = [idx for idx in range(len(xData)) if idx not in testIdx]

xTrain = [xData[i] for i in trainIdx]
yTrain = [yData[i] for i in trainIdx]
xTest = [xData[i] for i in testIdx]
yTest = [yData[i] for i in testIdx]


### 3. Fit the regression model using train set. (Not used SK-learn)
# (X^TX)^-1X^Ty : normal equation
trainXMatrix = np.matrix(xTrain)
trainYMatrix = np.matrix(yTrain).T
corrMatrix = trainXMatrix.T * trainXMatrix
# 역행렬의 유무를 체크한다.
if np.linalg.det(corrMatrix) == 0.0:
    raise ('This matrix is singular, cannot do inverse.')
# 정규방적식을 이용하여 coefficient의 값을 구한다.
wHat = np.linalg.inv(corrMatrix) * trainXMatrix.T * trainYMatrix


### 4. Test set의 mse를 구해본다.
testXMatrix = np.matrix(xTest)
testYMatrix = np.matrix(yTest).T
# mse
prediction = testXMatrix * wHat
error = prediction - testYMatrix
mse = sum([pow(error[i], 2) for i in range(len(error))]) / nTest


### 5. Using SK-learn
model = linear_model.LinearRegression()
model.fit(trainXMatrix, trainYMatrix)
# SK-learn 으로 얻은 mse
predictionSK = model.predict(testXMatrix)
errorSK = predictionSK - testYMatrix
mseSK = sum([pow(errorSK[i], 2) for i in range(len(error))]) / nTest

print 'MSE when using normal equation for LR : %f' %(mse)
print 'MSE when using SK-learn for LR : %f' %(mseSK)

yHat = [np.array(prediction)[i][0] for i in range(nTest)]
yHatSK = [np.array(predictionSK)[i][0] for i in range(nTest)]

print 'Correlation Coefficient using normal equation for LR : %f' %(np.corrcoef(yHat, yTest)[0][1])
print 'Correlation Coefficient using SK-learn for LR : %f' %(np.corrcoef(yHatSK, yTest)[0][1])