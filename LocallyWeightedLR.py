# -*- coding: utf-8 -*-

'''
일반적인 회귀분석은 underfit 경향이 크다. 평균 오류를 줄이는 방법 중 하나로, 지역적 가중치가 부여된 선형 회귀를 사용한다.
Locally weighted Linear regression
'''

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
nTrain = len(xTrain)

xTest = [xData[i] for i in testIdx]
yTest = [yData[i] for i in testIdx]


### 3. Fit the regression model using train set and weighted matrix. (Not used SK-learn)
# (X^TWX)^-1X^TWy : locally weighted normal equation
trainXMatrix = np.matrix(xTrain)
trainYMatrix = np.matrix(yTrain).T

testXMatrix = np.matrix(xTest)
testYMatrix = np.matrix(yTest).T

# test set의 data 하나씩을 이용하여 weighted matrix를 생성하고, 모델을 fit하고, test 한 값을 구한다.
k = 0.01
prediction = []
comp = 0
for testData in testXMatrix:
    comp += 1
    '''
    Gauss probability function.
    testData가 train set의 data와 가까우면 그 train data에 대해서는 높은 가중치를 부여하게 되고,
    그 weight matrix를 학습할 때 반영한다.
    결과로 얻은 학습 모델을 이용하여, 그 test data에 대한 prediction을 수행한다.
    즉, test data마다 weight matrix를 구하고, 그 때 구해진 모델로 학습을 수행한다. 그리고 예측한다.
    '''
    # weigth matrix 생성
    wMatrix = np.eye(nTrain)
    for j in range(nTrain):
        diff = trainXMatrix[j, :] - testData
        wMatrix[j][j] = np.exp((diff * diff.T) / -2*k**2)
    corrMatrix = trainXMatrix.T * wMatrix * trainXMatrix
    # 역행렬의 유무를 체크한다.
    if np.linalg.det(corrMatrix) == 0.0:
         raise ('This matrix is singular, cannot do inverse.')
    # 정규방적식을 이용하여 coefficient의 값을 구한다.
    wHat = np.linalg.inv(corrMatrix) * trainXMatrix.T * wMatrix * trainYMatrix
    prediction.append(testData * wHat)
    # just for completion
    print '========== %d ===========' %comp


### 4. Test set의 mse를 구해본다.
# mse
error = np.matrix(map(float, prediction)).T - testYMatrix
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

yHat = map(float, prediction)
yHatSK = map(float, predictionSK)

print 'Correlation Coefficient using normal equation for LR : %f' %(np.corrcoef(yHat, yTest)[0][1])
print 'Correlation Coefficient using SK-learn for LR : %f' %(np.corrcoef(yHatSK, yTest)[0][1])