---
layout: post
title: KT AIVLE SCHOOL 5기 5주차 | 딥러닝(Deep learning)(1)
description: KT AIVLE SCHOOL 5기 5주차에 진행한 딥러닝(Deep learning) 강의 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## 딥러닝 개념 익히기

> 모델링 : 파라미터를 잘 찾는것 → train error를 최소화하는 과정
> 
> 튜닝 : val error를 최소화하는 과정


### 가중치

- 최적의 Weight 찾는 방법
    - 최적의 모델 : 오차가 가장 적은 모델
- 가중치 조정
    - 조금씩 weight를 조정하며 오차가 줄어드는지 확인
    - 지정한 횟수 혹은 더 이상 오차가 줄지 않을 때까지 위의 단계 반복
    - 많은 머신러닝 알고리즘 및 딥러닝은 학습시킬 때 이런 방식으로 모델이 생성됨

### 학습 절차

- 어떤 정보 : node 혹은 뉴런(Neuron)
    - 어떤 정보에 알맞은 가중치와 절편을 찾아가는 과정
- `model.fit`하는 순간
    1. 가중치(**파라미터**)에 **초기값**을 할당 (랜덤으로)
    2. 예측 결과를 뽑는다
    3. **오차**를 계산(`loss`) → **forward propagation(순전파)**
    4. 오차를 줄이는 방향으로 가중치 조정(방향 : `optimizer`, 얼만큼 : `learning rate`(**중요**)) → **back propagation(역전파)**
    5. 다시 1단계로 올라가 반복 (`epoch`)
    
    > **하이퍼파라미터 : 머신러닝에서 사람이 개입할 여지**
    > 
    
    **[30번 조정하며 최적의 가중치 찾는 과정]**
    
    ![Untitled](/assets/img/blog/KT_AIVLE/week5/dl01.png)
    

## 딥러닝 모델링 : 회귀

### 딥러닝 과정 및 구조

- 딥러닝에서 스케일링은 필수
    - Normalization(정규화) : 모든 값의 범위를 0 ~ 1로 변환 → **보통 많이 사용**
    - Standardization(표준화) : 모든 값을 , 평균 = 0, 표준편차 = 1로 변환
- Process
    - **각 단계(task)는** 이전 단계의 output을 **input으로 받아** **처리**한 후 다음 단계로 **전달**
    - **공통의 목표를 달성하기 위해서 동작**
    - ex) 상품 기획 → 디자인 → 생산 → 물류 입고 → 매장 판매
- 딥러닝 구조
    - Input : 입력되는 x의 분석 단위(Layer 아님)
    - Hidden Layer
    - Output Layer
    - Output

### 딥러닝 코드

- `Dense`
    - `input_shape = ( , )` : **분석 단위**에 대한 shape
        - 1차원 : (feature수, ), 2차원 : (rows, cols)
    - `output` : 예측 결과가 1개 변수
- `Compile`
    - 선언된 모델에 대해 몇가지 설정을 한 후, 컴퓨터가 이해할 수 있는 형태로 변환하는 작업
    - loss function(오차 함수)
        - **오차 계산 무엇으로 할지 결정**
        - 회귀는 보통 mse
    - `optimizer`
        - **오차를 최소화 하도록 가중치 조절**
        - `optimizer = ‘adam’` : `learning_rate` 기본 값 = 0.001
        - `optimizer = Adam(learning_rate = 0.1)` : 옵션 값 조정 가능
- `learning_rate`
    - 적절하게 조절하는 것이 좋다
- 학습
    - `epochs` : 반복 횟수 → 전체 데이터를 몇 번 학습
    - `validation_split = 0.2` → 학습 데이터의 20%를 검증 데이터로 사용
- 학습 곡선
    - `.history`
        - 학습 수행 과정에 가중치가 업데이트 되면서 그 때 마다의 성능 측정하여 기록
        - 학습 시 계산된 오차 기록(가이드)
    - 바람직한 곡선
        - epoch가 증가하면서 loss가 큰 폭으로 축소
        - 점차 loss 감소 폭이 줄어들면서 감소
    - 들쑬 날쑥하면서 loss 감소 → learning_rate 줄이기
    - val_loss가 줄어들다가 다시 상승(과적합)
        - epoch와 learnig_rate 조절
- Hidden Layer
    - layer 여러개 : **리스트로 입력**
    - hidden layer
        - **input_shape는 첫번째 layer만 필요**
        - activation
            - 히든 레이어는 활성함수 필요
            - 보통 ‘relu’ 사용
    - output layer : 예측 결과가 1개
    - 활성화 함수(Activation Function)
        - 현재 레이어의 결과값을 다음 레이어(연결된 각 노드)로 어떻게 전달할지 결정 / 변환 해주는 함
        - 없으면 히든 레이어를 아무리 추가해도 그냥 선형회귀
        - Hidden Layer: 선형함수 → 비선형 (ReLU), Output Layer: 결과값 다른 값으로 변환 (주로 분류 모델에서 필요)
            - Sigmoid, tanh, ReLU(Hidden Layer 국룰)
    - 보통 점차 줄여간다
- 코드
    
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import *
    from sklearn.preprocessing import MinMaxScaler
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.backend import clear_session
    from tensorflow.keras.optimizers import Adam
    #from keras.optimizers import Adam : 버전에 따라 다르다
    
    def dl_history_plot(history):
        plt.figure(figsize=(10,6))
        plt.plot(history['loss'], label='train_err', marker = '.')
        plt.plot(history['val_loss'], label='val_err', marker = '.')
    
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid()
        plt.show()
    
    path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/boston.csv'
    data = pd.read_csv(path)
    
    target = 'medv'
    x = data.drop(target, axis = 1)
    y = data.loc[:, target]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state = 20)
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    
    nfeatures = x_train.shape[1]
    model3 = Sequential([  Dense(2, input_shape = (nfeatures,), activation = 'relu'),
                           Dense(1)   ])
    model3.summary()
    model3.compile( optimizer= Adam(learning_rate=0.08), loss = 'mse')
    hist = model3.fit(x_train, y_train, epochs = 50 , validation_split= .2, verbose = 0).history
    dl_history_plot(hist)
    
    pred3 = model3.predict(x_val)
    print(f'RMSE : {mean_squared_error(y_val, pred3, squared=False)}')
    print(f'MAE  : {mean_absolute_error(y_val, pred3)}')
    print(f'MAPE : {mean_absolute_percentage_error(y_val, pred3)}')
    ```


## Feature Representation

### Hidden Layer

- 연결
    - 모든 노드 간 연결(Fully Connected)
    - 연결 제어(Locally Connected)
- 학습
    - 오차를 계산하고 오차를 줄이기 위해 파라미터(가중치) 업데이트
    - 각 노드 별로 값 생성
- Hidden Layer 내부에서 발생한 일
    - 기존 데이터로 **새로운 특징(new feature)**를 만듦
    - 예측 값과 실제 값 사이의 오차를 최소화 해주는 **유익한 특징**일 것이다
    - 기존 데이터가 **새롭게 표현(Representation)**되는 **Feature Engineering**이 진행된 것
- Deep Learning → Representation Learning

## 딥러닝 모델링: 이진 분류

- 결과를 변환시켜줄 활성화 함수가 필요(시그모이드 함수)

### 활성 함수(Activation Function)

- node의 결과를 변환 시켜주는 역할
    - Hidden Layer
        - Activation Function : ReLU
        - 기능 : 좀 더 깊이 있는 학습을 시키려고
    - Output Layer
        - 회귀 : X
        - 이진 분류
            - Activation Function: sigmoid
            - 기능: 결과를 0, 1로 변환
        - 다중 분류
            - Activation Function: softmax
            - 기능: 각 범주에 대한 결과를 범주별 확률 값으로 변환

### Loss Function : binary_crossentropy

- 이진 분류 모델에서 사용되는 loss function
- y=1, y=0 일 때 각각 오차들의 평균

### 실습

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import RandomOverSampler

# 학습곡선 함수
def dl_history_plot(history):
    plt.figure(figsize=(10,6))
    plt.plot(history['loss'], label='train_err', marker = '.')
    plt.plot(history['val_loss'], label='val_err', marker = '.')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()
    
path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/Attrition_train_validation.CSV"
data = pd.read_csv(path)
data['Attrition'] = np.where(data['Attrition']=='Yes', 1, 0)

target = 'Attrition'
data.drop('EmployeeNumber', axis = 1, inplace = True)
x = data.drop(target, axis = 1)
y = data.loc[:, target]

dum_cols = ['BusinessTravel','Department','Education','EducationField','EnvironmentSatisfaction','Gender',
            'JobRole', 'JobInvolvement', 'JobSatisfaction', 'MaritalStatus', 'OverTime', 'RelationshipSatisfaction',
            'StockOptionLevel','WorkLifeBalance' ]

x = pd.get_dummies(x, columns = dum_cols ,drop_first = True)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 200, random_state = 2022)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

n = x_train.shape[1]
# 60, 60, 10, 5, 1 
clear_session()
model = Sequential([Dense(60, input_shape = (n, ), activation = 'relu'), 
                           Dense(60, activation = 'relu'), 
                            Dense(10, activation = 'relu'), 
                            Dense(5, activation = 'relu'),
                           Dense(1, activation = 'sigmoid')])
model.summary()

model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'binary_crossentropy')
hist = model.fit(x_train, y_train, epochs = 100, validation_split = 0.2, verbose = 0).history

dl_history_plot(hist)

pred = model.predict(x_val)
pred = np.where(pred >= 0.5, 1, 0)

print(confusion_matrix(y_val, pred))
print(classification_report(y_val, pred))

# resampling
ros = RandomOverSampler()
x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)
print(y_train_ros.value_counts(normalize = True))
print(y_train_ros.value_counts())
```

## 딥러닝 모델링: 다중 분류

### Output Layer

- Node 수: y의 범주수
- Softmax: 각 class 별(Output Node)로 예측한 값을, 하나의 확률 값으로 반환

### 다중 분류 모델링을 위한 전처리

- 다중 분류: y가 범주이고, 범주가 3개 이
- 방법 1: 정수 인코딩 +  sparse_categorical_crossentropy
    - y: Integer Encoding → class들을 0부터 시작하여 순차 증가하는 정수로 인코딩
    - `int_encoder.classes_` → 배열의 인덱스가 인코딩 된 범주
    - `loss='sparse_categorical_crossentropy’`
        - y는 인덱스로 사용됨 : 해당 인덱스의 예측 확률로 계산(-logy)
- 방법 2: y값 one-hot encoding 하고, `loss = ‘categorical_crossentropy’`
    - y: One-Hot Encoding
    - `loss = ‘categorical_crossentropy’`

### 실습

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.backend import clear_session
from tensorflow.keras.optimizers import Adam

path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/iris.csv"
data = pd.read_csv(path)
data['Species'] = data['Species'].map({'setosa':0, 'versicolor':1, 'virginica':2})
target = 'Species'
x = data.drop(target, axis = 1)
y = data.loc[:, target]

# 방법 1
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = .3, random_state = 20)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

nfeatures = x_train.shape[1] #num of columns

clear_session()
model = Sequential( Dense( 3 , input_shape = (nfeatures,), activation = 'softmax') )
model.summary()
model.compile(optimizer=Adam(learning_rate=0.1), loss= 'sparse_categorical_crossentropy')
history = model.fit(x_train, y_train, epochs = 50, validation_split=0.2).history
dl_history_plot(history)

pred = model.predict(x_val)
pred_1 = pred.argmax(axis=1)
print(confusion_matrix(y_val, pred_1))
print(classification_report(y_val, pred_1))

# 방법 2
from tensorflow.keras.utils import to_categorical
y_c = to_categorical(y.values, 3)
x_train, x_val, y_train, y_val = train_test_split(x, y_c, test_size = .3, random_state = 2022)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

nfeatures = x_train.shape[1] #num of columns
clear_session()
model = Sequential([Dense(3, input_shape = (nfeatures,), activation = 'softmax')])
model.summary()
model.compile(optimizer=Adam(learning_rate=0.1), loss='categorical_crossentropy')
history = model.fit(x_train, y_train, epochs = 100,
                    validation_split=0.2).history
dl_history_plot(history)

pred = model.predict(x_val)
pred_1 = pred.argmax(axis=1)
y_val_1 = y_val.argmax(axis=1)
print(confusion_matrix(y_val_1, pred_1))
print(classification_report(y_val_1, pred_1))
```