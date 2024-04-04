---
layout: post
title: 4주차 | 머신러닝(Machine learning)(1)
description: KT AIVLE SCHOOL 5기 4주차에 진행한 머신러닝(Machine learning) 강의 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## 전처리(리뷰)

```python
def preprocess():
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


## 머신러닝 소개

**(1) 머신러닝에 대한 이해**

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

**(2) 분류와 회귀**

- 모델링을 하기 전에 원하는 결과가 분류인지 회귀인지를 명확히 이해해야 함
- 회귀는 **연속적인 숫자**를 예측하는 것이고, 분류는 **범줏 값**을 예측하는 것
- 연속적인 숫자 확인 방법
    - 두 값 사이에 **중간값**이 의미가 있는 숫자인지
    - 또는 두 값에 대한 **연산 결과**가 의미가 있는 숫자인지 등
- 예측해야 할 값에 연속성이 있는지 확인하면 분류와 회귀를 쉽게 구분할 수 있음
    - 분류: A일까? B일까?, 회귀: 얼마나 많이?
- **분류와 회귀는 서로 다른 함수를 사용해 모델링을 하게 됨 (중요)**
    - **문제 유형을 정확히 파악 → 알고리즘과 평가 방법을 선택 → 관련된 함수를 사용해 모델링**
    

**(3) 미리 알아둘 용어**

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
- 과대적합 vs 과소적합
    - 과대적합
        - 학습 데이터에 대해서는 성능이 매우 좋은데, 평가 데이터에서 대해서는 성능이 매우 좋지 않은 경우
        - 학습 데이터에 대해서 만 잘 맞는 모델 → **실전에서 예측 성능이 좋지 않음**
    - 과소적합
        - 학습 데이터보다 평가 데이터에 대한 성능이 매우 좋거나, 모든 데이터에 대한 성능이 매우 안 좋은 경우
        - **모델이 너무 단순**하여 학습 데이터에 대해 적절히 훈련되지 않은 경우

**(4) 데이터 준비 과정**

- 머신러닝은 데이터에서 규칙을 찾고 그 규칙에 기반해 예측 하는 것 → 데이터 준비 과정이 필요!!!
- 충분히 학습하고 평가할 수 있는 좋은 데이터가 준비 되어야 좋은 모델을 기대할 수 있음
- 대상 데이터를 불러와 충분히 탐색하고 적절히 전처리 한 후 데이터 분리 과정을 수행
    
**(5) 실습**

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

## 성능 평가

### 회귀 모델 성능 평가

> 예측 값이 실제 값에 가까울 수록 좋은 모델 → 오차로 모델 성능 평가

**(1) 용어 정리**

> $$\Large y$$ : 실제값 → 실제 예측하고자 하는 값, 오차 : 이 값과 예측값의 차이\
> $$\Large \bar{y}$$ : 평균값 → 이미 존해하는 평균으로 예측한 값\
> $$\Large \hat{y}$$ : 예측값 → 새롭게 모델로 예측한 값, 평균값보다 얼마나 잘 예측했는지 확인

**(2) 회귀 평가 지표 정리**

> $$\Large MSE = \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}$$\
> $$\Large RMSE = \sqrt{MSE} = \sqrt{\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}}$$\
> $$\Large MAE = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{n}$$\
> $$\Large MAPE = \frac{\sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|}{n}$$
> > **위 값 모두 작을 수록 모델 성능이 좋다**

**(3) 오차를 보는 다양한 관점**

> $$\Large SST = \sum_{i=1}^{n} (y_i - \bar{y})^2$$ : **전체 오차**\
> $$\Large SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ : **회귀식이 잡아내지 못한 오차**\
> $$\Large SSR = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2$$ : **회귀식이 잡아낸 오차**\
> $$\Large SST = SSR + SSE$$

**(4) 결정 계수 (R-Squared)**
- **전체 오차 중에서 회귀식이 잡아낸 오차 비율**
- 모델의 **설명력**이라고도 부름 (값이 클수록 좋음)

> $$\Large R^{2} = \frac{SSR}{SST} = 1 -  \frac{SSE}{SST} = 1-\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}{\sum_{i=1}^{n}(y_{i}-\bar{y}_{i})^2}$$

**(5) 실습**
    
```python
# 1. 데이터 준비
target = 'medv' ## target 확인
X = data.drop(target, axis = 1) ## 데이터 분리
y = data.loc[:, target]
## 7:3으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# 2. 모델링
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. 회귀 성능 평가
print(f'MAE : {mean_absolute_error(y_test, y_pred)}')
print(f'MSE : {mean_squared_error(y_test, y_pred)}')
print(f'RMSE : {mean_squared_error(y_test, y_pred, squared = False)}')
print(f'MAPE : {mean_absolute_percentage_error(y_test, y_pred)}')
print(f'R2 : {r2_score(y_test, y_pred)}')
```
    

### 분류 모델 성능 평가

>- 분류 모델은 0인지 1인지 예측하는 것
>- 예측 값이 실제 값과 일치하는 값이 많을수록 좋은 모델 → 정확히 예측한 비율로 모델 성능 평가

**(1) Confusion Matrix(혼동 행렬)**
    
|  | 예측Negative(0) | 예측Positive(1) |
| --- | --- | --- |
| 실제Negative(0) | $$TN$$ | $$FP$$ |
| 실제Positive(1) | $$FN$$ | $$TP$$ |

- 용어 : **결과(T / F) + 예측 (P / N), 행렬의 역슬래시 값은 항상 정답**
- 정확도(Accuracy) = $$\Large \frac{TN + TP}{TN + FP + FN + TP}$$
    - 정분류율
    - 가장 직관적으로 모델 성능 확인 가능한 평가지표
- 정밀도(Precision) = $$\Large \frac{TP}{FP + TP}$$
    - 예측 관점
    - 정밀도가 낮을 경우 발생하는 문제
        - 암이 아닌데 암이라 하여 불필요한 치료 발생
- 재현율(Recall) = $$\Large \frac{TP}{FN + TP}$$
    - 실제 관점
    - 민감도(Sensitivity)라고도 부름
    - 재현율이 낮을 경우 발생하는 문제
        - 암인 사람에게 암이 아니라고 하는 경우
- 정밀도와 재현율은 기본적으로 **Positive**에 대해서 이야기
    - **Negative**에 대한 정밀도와 재현율도 의미를 가짐
- 특이도(Specificity) = $$\Large \frac{TN}{FP + TN}$$
    - 실제 Negative 중에서 Negative로 예측한 비율
    - 특이도가 낮을 경우 발생하는 문제
        - 암이 아닌데 암이라 했으니 불필요한 치료가 발생

**(2) F1-Score**
- 정밀도와 재현율의 조화평균
- 관점이 다른 경우 조화평균이 큰 의미를 가짐
- 정밀도와 재현율이 적절하게 요구될 때 사용
- F1-Score = $$\Large \frac{2 \times Precision \times Recall}{Precision + Recall}$$

**(3) 실습**
    
```python
# 1. 데이터 준비
target = 'ADMIT' # target 확인
x = data.drop(target, axis=1) # 데이터 분리
y = data.loc[:, target]

# 2. 모델링
model = KNeighborsClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 3.분류 성능 평가

print('Confusion Matrix\n', confusion_matrix(y_test, y_pred), '\n') # 성능 평가

## 혼동 행렬 시각화
plt.figure(figsize = (5, 3))
sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, cmap = 'Blues', cbar = False)
plt.show()

print(f'Accuracy : {accuracy_score(y_test, y_pred)}')
## 참고 : 두 수의 차이로 과대적합, 과소적합을 알 수 있음 (약간 과대적합)
print(f'평가 성능(정확도) : {model.score(x_test, y_test)}') 
print(f'학습 성능(정확도) : {model.score(x_train, y_train)}\n')

print('Precision : ', precision_score(y_test, y_pred)) # default : 1에 대한 정밀도
print('Precision : ', precision_score(y_test, y_pred, average = 'binary')) # default
print('Precision : ', precision_score(y_test, y_pred, average = None)) # 둘다 출력
print('Precision : ', precision_score(y_test, y_pred, average = 'macro')) # 평균
print('Precision : ', precision_score(y_test, y_pred, average = 'weighted'), '\n') # 가중치

print('Recall : ', recall_score(y_test, y_pred))
print('Recall : ', recall_score(y_test, y_pred, average = None), '\n') # 둘다 출력 기억하기

print('F1-Score : ', f1_score(y_test, y_pred))
print('F1-Score : ', f1_score(y_test, y_pred, average = None), '\n') # 둘다 출력 기억하기

print('Classification_report \n\n', classification_report(y_test, y_pred))
```

## 기본 알고리즘

### Linear Regression

**(1) 정의 및 특징**
- 최선의 회귀 모델 : **오차 합이 최소**가 되는 모델(오차 합이 최소가 되는 가중치, 편향을 찾기)
- 단순 회귀 : 독립 변수 하나가 종속 변수에 영향을 미치는 선형 회귀
    
    ```python
    # 회귀계수 확인
    print(model.coef_)
    print(model.intercept_)
    ```
    
- 다중 회귀 : 여러 독립 변수가 종속 변수에 영향을 미치는 선형 회귀
    
    ```python
    # 회귀계수 확인
    print(list(x_train))
    print(model.coef_)
    print(model.intercept_)
    ```
    
**(2) 실습**
    
```python
# 1. 데이터 준비
target = 'dist' # target 확인
x = data.drop(target, axis=1) # 데이터 분리
y = data.loc[:, target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# 2. 모델링
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('MAE : ', mean_absolute_error(y_test, y_pred))
print('R2 : ', r2_score(y_test, y_pred))
# 회귀계수 확인
print(model.coef_)
print(model.intercept_)

# 3. 기타
# 회귀식 만들기
a = model.coef_
b = model.intercept_
speed = np.linspace(x_test.min(), x_test.max(), 10)
dist = a * speed + b

# 회귀선 표시
dist_mean = y_train.mean()
plt.scatter(x_test, y_test) # 평가 데이터
plt.scatter(x_test, y_pred) # 학습 데이터
plt.plot(speed, dist, color = 'r') # 학습 데이터에 대한 예측값
plt.axhline(dist_mean, color = 'r', linestyle = '--')
plt.title('Speed & Distance', size = 20, pad = 10)
plt.xlabel('Speed(mph)')
plt.ylabel('Dist(ft)')
plt.show()

# 시각화
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.ylabel('Dist(ft)')
plt.show()
```
    
### K-Nearest Neighbor

**(1) 정의 및 특징**
- k개의 최근접 이웃의 값을 찾아 그 값들로 새로운 값을 예측하는 알고리즘
- 회귀와 분류에 사용되는 매우 간단한 지도 학습 알고리즘 → 연산 속도는 느림
    - k개 값의 평균으로 예측, 가장 많이 포함된 유형으로 분류
- k값의 중요성
    - **적절한 k값을 찾는 것이 중요(기본값 = 5)**
    - **일반적으로 1이 아닌 홀수로 설정**
- 거리 구하기
    - 맨하튼 거리(두 지점의 각 좌표의 차의 절대값) ≥ 유클리드 거리(두 지점의 거리)
- Scaling 필요성
    - 스케일링 여부에 따라 KNN 모델 성능이 달라질 수 있음
    - 대표적인 스케일링: **정규화(Normalization), 표준화(Standardization)**
    - 평가용 데이터에도 **학습용 데이터** **기준**으로 스케일링 수행
        
        **[참고] 학습 데이터를 기준으로 정규화**
        
        <img src = 'https://github.com/Jangrae/img/blob/master/minmax.png?raw=true'>
        
**(2) 실습**
    
```python
# 1. 데이터 준비
data.interpolate(method='linear', inplace=True) # 결측치 채우기

#  변수 제거
drop_cols = ['Month', 'Day'] 
data.drop(drop_cols, axis=1, inplace=True)

target = 'Ozone' # target 확인

# 데이터 분리
x = data.drop(target, axis=1)
y = data.loc[:, target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# 정규화
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델링
model = KNeighborsRegressor() # n_neighbors를 데이터의 수로 하면 모델이 평균값이 된다
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('MAE : ', mean_absolute_error(y_test, y_pred))
print('R2 : ', r2_score(y_test, y_pred))

# 3. 기타
# 예측값, 실젯값 시각화
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.ylabel('Ozone')
plt.show()
```
    

### Decision Tree

**(1) 정의 및 특징**
- 특정 변수에 대한 의사결정 규칙을 **나무 가지**가 뻗는 형태로 분류해 나감
- **분류**와 **회귀** 모두에 사용되는 지도학습 알고리즘
- 분석 과정을 실제로 확인 가능 → **화이트 박스 모델**
- **의미 있는 질문**을 먼저 하는 것이 중요
- **과적합** 발생하기 쉬움
    - **트리 깊이를 제한**하는 튜닝이 필요
- Root Node(뿌리 마디), Terminal Node(끝 마디, Leaf Node), Depth(깊이)
- 분류와 회귀
    - 비용함수: 분류 → **불순도**, 회귀 → **MSE**
    - 분류: 마지막 노드에 있는 샘플들의 **최빈값**을 예측값으로 반환
    - 회귀: 마지막 노드에 있는 샘플들의 **평균**을 예측값으로 반환

**(2) 불순도(Impurity)**
- 불순도가 낮을 수록 분류가 잘 된 것
- 불순도 수치화 가능한 지표
    - 지니 불순도(Gini Impurity)
        > $$- (양성 클래스 비율^2 + 음성 클래스 비율^2)$$
        
        - 분류 후 얼마나 잘 분류했는지 평가하는 지표
        - 특징
            - 지니 불순도가 낮을수록 순도가 높음
            - 0 ~ 0.5 사이의 값(이진 분류의 경우) → 순수하게 분류: 0, 완벽하게 섞이면: 0.5
        - 지니 불순도가 낮은 속성으로 의사결정 트리 노드 결정
    - 엔트로피(Entropy)
        > $$- 음성클래스비율 \times log_2(음성 클래스 비율) - 양성클래스비율 \times log_2(양성 클래스 비율)$$
        
        - $$p_i$$ : 집합 안에서 속성 i의 확률
        - 0 ~ 1사이의 값 → 순수하게 분류되면: 0, 완벽하게 섞이면: 1
        - 정보 이득(Information Gain)
            > $$𝐺𝑎𝑖𝑛 (𝑇, 𝑋) = 𝐸𝑛𝑡𝑟𝑜𝑝𝑦 (𝑇) − 𝐸𝑛𝑡𝑟𝑜𝑝𝑦(𝑇, 𝑋)$$
            
            - 정보 이득이 크다 = 어떤 속성으로 분할할 때 불순도가 줄어든다
            - 정보 이득이 가장 큰 속성부터 분할

**(3) 가지치기**
- 가지치기를 하지 않으면 → 과대적합, 일반화되지 못함
- 여러 하이퍼파라미터 값을 조정해 가지치기 할 수 있음
    - **max_depth(트리의 최대 깊이(기본값: None))**
    - **min_samples_leaf(노드를 분할하기 위한 최소한의 샘플 개수(기본값: 2))**
    - **min_samples_split(리프 노드가 되기 위한 최소한의 샘플 수(기본값: 1))**
- 가장 적절한 하이퍼파라미터 값 찾도록 노력해야 함
- Decision Tree 분류 모델도 결국 **확률**에 근거해 **예측**을 한다

**(4) 실습**
    
```python
# 1. 데이터 준비
# 제거 대상: PassengerId, Name, Ticket, Cabin
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
# 변수 제거
data.drop(drop_cols, axis=1, inplace=True)
# Age 결측치를 중앙값으로 채우기
age_median = data['Age'].median()
data['Age'].fillna(age_median, inplace=True)
# Embarked 최빈값으로 채우기
emb_freq = data['Embarked'].mode()[0]
data['Embarked'].fillna(emb_freq, inplace=True)

# target 확인
target = 'Survived'
# 데이터 분리
x = data.drop(target, axis=1)
y = data.loc[:, target]
# 가변수화 대상: Pclass, Sex, Embarked
dumm_cols = ['Pclass', 'Sex', 'Embarked']
# 가변수화
x = pd.get_dummies(x, columns=dumm_cols, drop_first=True, dtype=int)
# 7:3으로 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# 2. 모델링
model = DecisionTreeClassifier(max_depth = 5, random_state=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(confusion_matrix(y_test, y_pred), '\n')
print(classification_report(y_test, y_pred))

# 3. 기타
# 이미지 파일 만들기
export_graphviz(model,                                 # 모델 이름
                out_file='tree.dot',                   # 파일 이름
                feature_names=x.columns,               # Feature 이름
                class_names=['die', 'survived'],       # Target Class 이름
                rounded=True,                          # 둥근 테두리
                precision=2,                           # 불순도 소숫점 자리수
                # max_depth = 3,                         # 표시할 트리 깊이
                filled=True)                           # 박스 내부 채우기

# 파일 변환
!dot tree.dot -Tpng -otree.png -Gdpi=300
# 이미지 파일 표시
Image(filename='tree.png')

# 변수 중요도 데이터프레임 만들기
df = pd.DataFrame()
df['feature'], df['importance'] = list(x), model.feature_importances_
df.sort_values(by='importance', ascending=True, inplace=True)
# 시각화
plt.figure(figsize=(5, 5))
plt.barh(df['feature'], df['importance']) # 오름차순으로 정렬해야 내림차순으로 보여진다
plt.show()
```

### Logistic Regression

**(1) 정의와 특징**
- 로지스틱 회귀: 확률 문제를 선형회귀로 모델링
- 로지스틱 함수
    - **시그모이드 함수** : $$\Large p = \frac {1} {1 + e^{-f(x)}}$$
    - (-∞, ∞) 범위를 갖는 선형 판별식 결과로 (0, 1) 범위의 확률 값을 얻게 됨
    - **학습 데이터를 잘 설명하는 선형 판별식의 기울기(𝑎)와 절편(𝑏)을 찾는 문제**

**(2) 실습**
    
```python
# 1.데이터 준비
target = 'Outcome' # target 확인
# 데이터 분리
x = data.drop(target, axis=1)
y = data.loc[:, target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# 2. 모델링
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 3. 기타
# 예측값 확인
print(y_test.values[10:30])
print(y_pred[10:30])

# 확률값 확인
p = model.predict_proba(x_test)
print(p[10:30])

# 1의 확률값 얻기
p1 = p[:, [1]]
# 임계값 0.5
y_pred2 = np.array([1 if x > 0.5 else 0 for x in p1])
print(y_pred2[:20])
print(classification_report(y_test, y_pred2))
# 임계값 0.45
y_pred2 = np.array([1 if x > 0.45 else 0 for x in p1])
print(y_pred2[:20])
print(classification_report(y_test, y_pred2))
```