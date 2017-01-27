# -*- coding: utf-8 -*-

'''
classifier의 성능을 측정하는 방법
1. 오분류 오차 측정
2. ROC 곡선

<ROC 곡선>
 - Receiver operating characteristic
 - classifier의 전체적인 성능을 특성화 할 수 있는 기법
 - 사용하는 검사법 및 classifier의 기준치(cut off value)를 결정하려고 할 때 사용하는 방법
 - 주로 의학 및 역학에서 많이 사용하는 방법
 - 민감도(sensitivity), 1-특이도(specificity)를 각각 y축, x축으로 하는 곡
   - 민감도 : 예를 들어, 질병을 가지고 있는 사람중에서, 검사 결과가 양성이 나오는 비율 (True positive ratio)
   - 특이도 : 예를 들어, 질병을 가지고 있지 않은 사람중에서, 검사 결과가 음성이 나오는 비율 (1-False positive ratio)
   - True positive ratio : TP / (TP + FN)
   - False positive ratio : FP / (FP + TN)
<AUC>
 - Area under the curve
 - ROC curve의 영역.
 - 0.5와 1사이

<Detail of implementation>
(1) feature 중, 바위와 기뢰를 각각 1,0으로 labeling 한다.
(2) training set과 test set의 비율은 전체 데이터를 기준으로 7:3의 비율로 나눌 것이다.
(3) training 은, SK-learn의 Ridge linear regression을 사용할 것이다. 중요한 것은 classifier로 활용할 것이다.
(4) training 후, 모델의 성능을 측정하기 위하여 AUC를 사용할 것이다. (Root mean squared error)
'''

__author__ = "Jeonghun Yoon"

from urllib2 import urlopen
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plot


##############################################
# (1) 데이터를 load 한다. label data를 float로 변환
##############################################
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
data = urlopen(url)
x_data = []
y_data = []

for line in data.readlines():
    tokens = line.strip().split(",")
    # label 데이터
    if tokens[-1] is "M":
        y_data.append(1.0)
    else:
        y_data.append(0.0)
    del (tokens[-1])
    # feature vector
    x_data.append(map(float, tokens))


#############################################
# (2) training set과 test set의 비율은 7:3 이다.
#############################################
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)


##################################################################
# (3) 모델을 training 한다. Ridge regression을 이용하여 classify를 한다.
##################################################################
alpha_list = [0.1**i for i in [-3, -2, -1, 0, 1, 2, 3, 4, 5]]
model_list = []

for alpha in alpha_list:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(x_train, y_train)
    model_list.append(ridge_model)


################################
# (4) 성능 측정을 위하여 AUC를 구한다.
################################
auc_list = []
for model in model_list:
    fpr, tpr, thresholds = roc_curve(y_test, model.predict(x_test))
    auc_list.append(auc(fpr, tpr))


#################
# (5) AUC의 시각화
#################
for i in range(len(alpha_list)):
    print("alpha : ", alpha_list[i], ", auc : ", auc_list[i])

plot.figure()
plot.plot([-3, -2, -1, 0, 1, 2, 3, 4, 5], auc_list)
plot.xlabel("-log(alpha)")
plot.ylabel("AUC")
plot.show()