---
layout: post
title: KT AIVLE SCHOOL 5기 4주차 | 머신러닝(Machine learning)(1)
description: KT AIVLE SCHOOL 5기 4주차에 진행한 머신러닝(Machine learning) 강의 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## 0. 전처리(리뷰)

```python
# NaN 열이 포함된 모든 변수(axis=1) 제거
titanic.dropna(axis=1, inplace=True)

# NaN을 평균값으로 채우기
titanic['Age'].fillna(mean_age, inplace=True)

# NaN 값을 가장 빈도가 높은 값으로 채우기
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Ozone 변수 NaN 값을 바로 앞의 값으로 채우기
air['Ozone'].fillna(method='ffill', inplace=True)

# Solar.R 변수 NaN 값을 바로 뒤의 값으로 채우기
air['Solar.R'].fillna(method='bfill', inplace=True)

# 선형 보간법으로 채우기
air['Ozone'].interpolate(method='linear', inplace=True)

# 가변수화 - drop_first = True 로 다중공선성 문제 제거 가능
titanic = pd.get_dummies(titanic, columns=dumm_cols, drop_first=True, dtype = int)
```


## 1. 머신러닝 소개

### 머신러닝에 대한 이해

- `인간의 경험 = 머신의 데이터`
    - **적당량의 학습 데이터를 주는 것이 중요**
- 학습 방법에 따른 분류
    - 지도 학습 : 데이터의 패턴을 배우게 하는 학습 방법
    - 비지도 학습 : 정답이 없는 데이터 만으로 배우게 하는 학습 방법
    - 강화 학습 : 선택한 결과에 대해 보상을 받아 행동을 개선하면서 배우게 하는 학습 방법
- 과제에 따른 분류
    - 분류 문제 : 이미 적절히 분류된 데이터를 통해 규칙을 찾아 그 규칙을 기반으로 새롭게 주어진 데이터를 적절히 분류하는 것이 목적 **(지도 학습)**
    - 회귀 문제 : 이미 결과 값이 있는 데이터를 통해 입력값과 출력값의 연관성을 찾아 그 연관성을 기반으로 새롭게 주어진 데이터에 대한 값을 예측하는 것이 목적 **(지도 학습)**
    - 클러스터링 : 주어진 데이터를 학습하여 적절한 분류 규칙을 찾아 데이터를 분류함을 목적으로 함, 정답이 없으니 성능을 평가하기 어려움 **(비지도 학습)**

### 분류와 회귀

- 모델링을 하기 전에 원하는 결과가 분류인지 회귀인지를 명확히 이해해야 함
- 회귀는 **연속적인 숫자**를 예측하는 것이고, 분류는 **범줏 값**을 예측하는 것
- 연속적인 숫자 확인 방법
    - 두 값 사이에 **중간값**이 의미가 있는 숫자인지
    - 또는 두 값에 대한 **연산 결과**가 의미가 있는 숫자인지 등
- 예측해야 할 값에 연속성이 있는지 확인하면 분류와 회귀를 쉽게 구분할 수 있음
    - 분류: A일까? B일까?, 회귀: 얼마나 많이?
- **분류와 회귀는 서로 다른 함수를 사용해 모델링을 하게 됨 (중요)**
    - **문제 유형을 정확히 파악 → 알고리즘과 평가 방법을 선택 → 관련된 함수를 사용해 모델링**
    

### 미리 알아둘 용어

- 모델, 모델링
    - 모델 : 데이터로부터 패턴을 찾아 수식으로 정리해 놓은 것
    - 모델링 : 오차가 적고 성능이 좋은 모델을 만드는 과정
    - 모델의 목적
        - 샘플을 가지고 전체를 추정
- 행, 열
    - 행 : 개체, **관측치**, 기록, 사례, 경우
    - 열 : 특성, 속성, **변수**, 필드
- 독립변수, 종속변수
    - 원인과 결과 (x, y)
- 오차
    - 평균과 오차
        - 통계학에서 사용되는 **가장 단순한 모델** 중 하나 : **평균**
        - 관측값(=실젯값)과 모델 예측값의 차이: 이탈도(Deviance) → **오차**
- 데이터 분리
    - 실전 : 학습용, 검증용, 평가용 데이터로 분리
        - 평가용 : 별도 제공 데이터일 경우 많음
        - 검증용 : 평가 전에 모델 성능 검증 가능 (튜닝 시 사용)
    - 수업 : 편의상 모델 생성 후 평가용으로 바로 평가
    
    ```python
    # x, y 분리
    target = 'Ozone'
    x = data.drop(target, axis=1)
    y = data.loc[:, target]
    ```
    
    ```python
    # 학습용, 평가용 데이터 분리
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 
    ```
    
- 과대적합 vs 과소적합
    - 과대적합
        - 학습 데이터에 대해서는 성능이 매우 좋은데, 평가 데이터에서 대해서는 성능이 매우 좋지 않은 경우
        - 학습 데이터에 대해서 만 잘 맞는 모델 → **실전에서 예측 성능이 좋지 않음**
    - 과소적합
        - 학습 데이터보다 평가 데이터에 대한 성능이 매우 좋거나, 모든 데이터에 대한 성능이 매우 안 좋은 경우
        - **모델이 너무 단순**하여 학습 데이터에 대해 적절히 훈련되지 않은 경우

### 모델링 코드 구조

- Scikit-Learn 소개
    - 지도/비지도 학습 알고리즘을 제공하는 대표적인 파이썬 라이브러리
    - **sklearn**이라고 쓰고 '**사이킷런**' 이라고 읽음
    - **여러 알고리즘을 같은 구조**의 코드로 사용할 수 있어 **배우기 쉬움**
    - Iris, Boston과 같은 다양한 예제 데이터셋(토이 데이터셋)을 포함하고 있어 학습에 용이
- 데이터 준비 과정
    - 머신러닝은 데이터에서 규칙을 찾고 그 규칙에 기반해 예측 하는 것 → 데이터 준비 과정이 필요!!!
    - 충분히 학습하고 평가할 수 있는 좋은 데이터가 준비 되어야 좋은 모델을 기대할 수 있음
    - 대상 데이터를 불러와 충분히 탐색하고 적절히 전처리 한 후 데이터 분리 과정을 수행
    
    ```python
    # 라이브러리 불러오기
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    
    # 데이터 읽어오기
    data = pd.read_csv('airquality.csv')
    
    # x, y 분리
    target = 'Ozone'
    x = data.drop(target, axis=1)
    y = data.loc[:, target]
    
    # 학습용, 평가용 데이터 분리
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 
    ```
    
- 모델링 과정
    
    ```python
    # 1. 불러오기
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    
    # 2. 선언하기
    model = LinearRegression()
    
    # 3. 학습하기
    model.fit(x_train, y_train)
    
    # 4. 예측하기
    y_pred = model.predict(x_test)
    
    # 5. 평가하기
    mean_absolute_error(y_test, y_pred)
    ```
    
- 머신러닝 학습 방법(권고)
    - 분류와 회귀 이해 → 모델링 코드 익히기 → 평가 방법 이해 → 중요 알고리즘 원리 이해
        - 우선 사용하고, 그 다음에 배우기!

### 실습

```python
# 상관관계 시각화
sns.heatmap(data.corr(), annot = True, cmap = 'Blues', cbar = False,
            square = True, fmt = '.2f', annot_kws = {'size' : 9})
plt.show()

# 예측값과 실젯값 시각화 비교
medv_mean = y_train.mean()
print(f'평균 : {medv_mean}')

plt.plot(y_test.values, label = 'Actual')
plt.plot(y_pred, label = 'Predicted')
plt.legend()
plt.axhline(medv_mean, color = 'r')
plt.show()

# GPA --> ADMIT (수치형 변수에 따른 범주형 변수(target)의 분포 확인)
sns.histplot(x = data['GPA'], hue = data['ADMIT'], bins = 30)
plt.show()

# 7:3으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 1, shuffle = True, stratify = y) 
# 시계열 데이터에서는 shuffle 금지(default: True), stratify : 라벨 별로 균등하게 분포되도록 해주는 옵션
```

## 2.  성능 평가

### 회귀 모델 성능 평가

> 예측 값이 실제 값에 가까울 수록 좋은 모델 → 오차로 모델 성능 평가

#### 용어 정리
> $$y$$ : 실제값 → 실제 예측하고자 하는 값, 오차 : 이 값과 예측값의 차이\
> $$\bar{y}$$ : 평균값 → 이미 존해하는 평균으로 예측한 값\
> $$\hat{y}$$ : 예측값 → 새롭게 모델로 예측한 값, 평균값보다 얼마나 잘 예측했는지 확인

#### 회귀 평가 지표 정리
> $$MSE = \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}$$\
> $$RMSE = \sqrt{MSE} = \sqrt{\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}}$$\
> $$MAE = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{n}$$\
> $$MAPE = \frac{\sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|}{n}$$\
> > 위 값 모두 작을 수록 모델 성능이 좋다

#### 오차를 보는 다양한 관점
> $$SST = \sum_{i=1}^{n} (y_i - \bar{y})^2$$ : 전체 오차\
> $$SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ : 회귀식이 잡아내지 못한 오차\
> $$SSR = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2$$ : 회귀식이 잡아낸 오차\
> $$SST = SSR + SSE$$

#### 결정 계수 (R-Squared)
- **전체 오차 중에서 회귀식이 잡아낸 오차 비율**
- 모델의 **설명력**이라고도 부름 (값이 클수록 좋음)

> $$R^{2} = \frac{SSR}{SST} = 1 -  \frac{SSE}{SST} = 1-\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}{\sum_{i=1}^{n}(y_{i}-\bar{y}_{i})^2}$$

#### 실습
    
```python
# 1. 환경 준비
## 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')
%config InlineBackend.figure_format = 'retina'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

# 2. 데이터 이해
## 상위 몇 개 행 확인
data.head()
## 변수 확인
data.info()
## 기술통계 확인
data.describe().T
## 상관관계 확인
corr_matrix = data.corr()
corr_matrix
## 상관관계 시각화
sns.heatmap(corr_matrix, annot = True, cmap = 'Blues', cbar = False, square = True,
            fmt = '.2f', annot_kws = {'size' : 8})
plt.show()
## 산점도
plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
sns.regplot(x = data['lstat'], y = data['medv'])
plt.subplot(1, 2, 2)
sns.regplot(x = data['rm'], y = data['medv'])
plt.show()

# 3. 데이터 준비
## target 확인
target = 'medv'
## 데이터 분리
X = data.drop(target, axis = 1)
y = data.loc[:, target]
## 7:3으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# 4. 모델링
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. 회귀 성능 평가
print(f'MAE : {mean_absolute_error(y_test, y_pred)}')
print(f'MSE : {mean_squared_error(y_test, y_pred)}')
print(f'RMSE : {mean_squared_error(y_test, y_pred, squared = False)}')
print(f'MAPE : {mean_absolute_percentage_error(y_test, y_pred)}')
print(f'R2 : {r2_score(y_test, y_pred)}')
```
    

### 분류 모델 성능 평가

>- 분류 모델은 0인지 1인지 예측하는 것
>- 예측 값이 실제 값과 일치하는 값이 많을수록 좋은 모델 → 정확히 예측한 비율로 모델 성능 평가

#### Confusion Matrix(혼동 행렬)
    
|  | 예측Negative(0) | 예측Positive(1) |
| --- | --- | --- |
| 실제Negative(0) | TN | FP |
| 실제Positive(1) | FN | TP |

- 용어 : **결과(T / F) + 예측 (P / N), 행렬의 역슬래시 값은 항상 정답**
- 정확도(Accuracy) = $$\frac{TN + TP}{TN + FP + FN + TP}$$
    - 정분류율
    - 가장 직관적으로 모델 성능 확인 가능한 평가지표
- 정밀도(Precision) = $$\frac{TP}{FP + TP}$$
    - 예측 관점
    - 정밀도가 낮을 경우 발생하는 문제
        - 암이 아닌데 암이라 하여 불필요한 치료 발생
- 재현율(Recall) = $$\frac{TP}{FN + TP}$$
    - 실제 관점
    - 민감도(Sensitivity)라고도 부름
    - 재현율이 낮을 경우 발생하는 문제
        - 암인 사람에게 암이 아니라고 하는 경우
- 정밀도와 재현율은 기본적으로 **Positive**에 대해서 이야기
    - **Negative**에 대한 정밀도와 재현율도 의미를 가짐
- 특이도(Specificity) = $$\frac{TN}{FP + TN}$$
    - 실제 Negative 중에서 Negative로 예측한 비율
    - 특이도가 낮을 경우 발생하는 문제
        - 암이 아닌데 암이라 했으니 불필요한 치료가 발생
#### F1-Score
- 정밀도와 재현율의 조화평균
- 관점이 다른 경우 조화평균이 큰 의미를 가짐
- 정밀도와 재현율이 적절하게 요구될 때 사용
- F1-Score = $$\frac{2 \times Precision \times Recall}{Precision + Recall}$$

