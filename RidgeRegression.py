#-*- coding: utf-8 -*-

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



