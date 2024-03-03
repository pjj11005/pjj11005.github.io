---
layout: post
title: KT AIVLE SCHOOL 5기 2주차 | 데이터 분석(Data analysis)(3)
description: KT AIVLE SCHOOL 5기 2주차에 진행한 데이터 분석(Data analysis) 강의 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}


## 9. 종합실습2 이변량분석(y-범주) : 직원 이직 분석

![attrition](https://www.talentalign.com/wp-content/uploads/2019/09/main-qimg-2434943004743f79c0abf540cb4f6e2c-cropped.jpg)

* 직원 이직 분석
    * 회사에서 최근 1~2년 사이 이직률이 상승했다
    * 직원들이 이직하는데 중요한 요인이 무엇인지 데이터를 기반으로 한 분석을 의뢰 받았다

### 1.환경준비

* 라이브러리 불러오기


```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic      #mosaic plot!

import scipy.stats as spst
```

* 데이터 불러오기 : 다음의 예제 데이터를 사용합니다.


```python
# 직원 이직 데이터
path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/Attrition_simple3.csv'
data = pd.read_csv(path)
data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attrition</th>
      <th>Age</th>
      <th>DistanceFromHome</th>
      <th>EmployeeNumber</th>
      <th>Gender</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>OverTime</th>
      <th>PercentSalaryHike</th>
      <th>TotalWorkingYears</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>27</td>
      <td>2</td>
      <td>1898</td>
      <td>Female</td>
      <td>3</td>
      <td>Single</td>
      <td>6500</td>
      <td>No</td>
      <td>14</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>27</td>
      <td>9</td>
      <td>1965</td>
      <td>Male</td>
      <td>4</td>
      <td>Single</td>
      <td>4105</td>
      <td>No</td>
      <td>14</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>44</td>
      <td>2</td>
      <td>1703</td>
      <td>Female</td>
      <td>4</td>
      <td>Married</td>
      <td>7879</td>
      <td>Yes</td>
      <td>19</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>42</td>
      <td>2</td>
      <td>1231</td>
      <td>Male</td>
      <td>1</td>
      <td>Married</td>
      <td>6272</td>
      <td>No</td>
      <td>16</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>32</td>
      <td>1</td>
      <td>2016</td>
      <td>Female</td>
      <td>4</td>
      <td>Married</td>
      <td>10422</td>
      <td>No</td>
      <td>19</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



> * 변수설명
    * **Attrition**	:	이직여부, Yes(1) , No(0) (**Target**)
    * **Age**	:	나이
    * **DistanceFromHome**	:	집-직장 거리(마일)
    * **EmployeeNumber**	:	사번
    * **Gender**	:	성별(Male, Female)
    * **JobSatisfaction**	:	직무 만족도 (1 ~ 4)
    * **MaritalStatus**	:	결혼상태(Married, Single, Divorced)
    * **MonthlyIncome**	:	월급(달러)
    * **OverTime**	:	야근여부
    * **PercentSalaryHike**	:	전년대비 급여인상율(%)
    * **TotalWorkingYears**	:	총 근무 연수



### 2. 단변량 분석

#### (1) Attrition


```python
# Attrition
target = 'Attrition'
data[target].value_counts()
```




    Attrition
    0    405
    1    195
    Name: count, dtype: int64




```python
data[target].value_counts(normalize = True)
```




    Attrition
    0    0.675
    1    0.325
    Name: proportion, dtype: float64




```python
sns.countplot(x = target, data = data)
plt.grid()
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_13_0.png)
    


- 이직한 직원의 비율 32.5%

#### (2) OverTime


```python
var = 'OverTime'
data[var].value_counts()
```




    OverTime
    No     405
    Yes    195
    Name: count, dtype: int64




```python
data[var].value_counts(normalize = True)
```




    OverTime
    No     0.675
    Yes    0.325
    Name: proportion, dtype: float64




```python
sns.countplot(x = var, data = data)
plt.grid()
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_18_0.png)
    


#### (3) PercentSalaryHike


```python
var = 'PercentSalaryHike'
data[[var]].describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PercentSalaryHike</th>
      <td>600.0</td>
      <td>15.288333</td>
      <td>3.667098</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>18.0</td>
      <td>25.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plt.figure(figsize =(8, 6))

plt.subplot(2, 1, 1)
sns.histplot(x = var, data = data, bins = 20, kde = True)
plt.grid()

plt.subplot(2, 1, 2)
sns.boxplot(x = var, data = data)
plt.grid()

plt.tight_layout()
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_21_0.png)
    


### 3.범주→범주


```python
def eda_cat_cat(feature, target, data):
    # 먼저 집계
    table = pd.crosstab(data[target], data[feature])
    table_norm = pd.crosstab(data[target], data[feature], normalize = 'columns')
    print('교차표\n', table)
    print('-' * 100)
    print('교차표 (norm = columns) \n', table_norm)
    print('-' * 100)
    
    mosaic(data, [feature, target])
    plt.axhline(1- data[target].mean(), color = 'r')
    plt.show()
    
    # 카이제곱검정
    result = spst.chi2_contingency(table)
    print('-' * 100)
    print('카이제곱통계량', result[0])
    print('p-value', result[1])
    print('자유도', result[2])
    print('기대빈도\n',result[3])
```

#### (1) Gender → Attrition

1) 교차표, mosaic, 카이제곱 검정


```python
target = 'Attrition'
feature = 'Gender'
    
eda_cat_cat(feature, target, data)
```

    교차표
     Gender     Female  Male
    Attrition              
    0             157   248
    1              66   129
    ----------------------------------------------------------------------------------------------------
    교차표 (norm = columns) 
     Gender       Female      Male
    Attrition                    
    0          0.704036  0.657825
    1          0.295964  0.342175
    ----------------------------------------------------------------------------------------------------
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_26_1.png)
    


    ----------------------------------------------------------------------------------------------------
    카이제곱통계량 1.1614318259891623
    p-value 0.28116879016055174
    자유도 1
    기대빈도
     [[150.525 254.475]
     [ 72.475 122.525]]
    

2) 파악된 내용

- 카이제곱검정으로는 관련이 없다고 나오나, 그래프로 볼때 약간 관련이 있다고 판단됨
- 남자의 이직률이 미세하게 여자보다 높다

#### (2) JobSatisfaction → Attrition

1) 교차표, mosaic, 카이제곱 검정

```python
feature = 'JobSatisfaction'
    
eda_cat_cat(feature, target, data)
```

    교차표
     JobSatisfaction   1   2    3    4
    Attrition                        
    0                74  79  114  138
    1                52  37   59   47
    ----------------------------------------------------------------------------------------------------
    교차표 (norm = columns) 
     JobSatisfaction         1         2        3         4
    Attrition                                             
    0                0.587302  0.681034  0.65896  0.745946
    1                0.412698  0.318966  0.34104  0.254054
    ----------------------------------------------------------------------------------------------------
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_30_1.png)
    


    ----------------------------------------------------------------------------------------------------
    카이제곱통계량 8.884191097554549
    p-value 0.03087092125625072
    자유도 3
    기대빈도
     [[ 85.05   78.3   116.775 124.875]
     [ 40.95   37.7    56.225  60.125]]
    

2) 파악된 내용

- 그래프로 볼 때, 만족도와 이직여부는 관련이 약간 있다
- 카이제곱통계량으로 볼 때도 약간의 관련성이 있다
- 미세하게 만족도 3인 직원들이 2인 직원들보다 이직율이 높다
    - 만족도가 아닌 다른 요인이 작용한 것 같다
- 만족도가 크다고 이직율이 현저히 낮지는 않다 (만족도 1과 4의 이직율 16% 차이)
* 만족도의 범위를 넓게 혹은 좁게해서 새롭게 구간을 나눠도 좋을것 같다

#### (3) MaritalStatus → Attrition

1) 교차표, mosaic, 카이제곱 검정

```python
feature = 'MaritalStatus'
    
eda_cat_cat(feature, target, data)
```

    교차표
     MaritalStatus  Divorced  Married  Single
    Attrition                               
    0                    89      205     111
    1                    23       69     103
    ----------------------------------------------------------------------------------------------------
    교차표 (norm = columns) 
     MaritalStatus  Divorced   Married    Single
    Attrition                                  
    0              0.794643  0.748175  0.518692
    1              0.205357  0.251825  0.481308
    ----------------------------------------------------------------------------------------------------
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_34_1.png)
    


    ----------------------------------------------------------------------------------------------------
    카이제곱통계량 37.829711907070525
    p-value 6.100738829354226e-09
    자유도 2
    기대빈도
     [[ 75.6  184.95 144.45]
     [ 36.4   89.05  69.55]]
    

2) 파악된 내용

- 그래프와 카이제곱 통계량으로 볼 때 관련이 꽤 있다
- 독신인 사람이 가장 이직율이 높다
    - 여유롭다
    - 야근 여부와의 관련성 분석 필요
- 이혼한 사람의 이직율이 가장 낮다, 결혼한 사람의 이직율도 낮다
    - 금전적 여유 부족, 안정적인 삶을 원하는 경향이 크다

#### (4) OverTime → Attrition

1) 교차표, mosaic, 카이제곱 검정

```python
feature = 'OverTime'
    
eda_cat_cat(feature, target, data)
```

    교차표
     OverTime    No  Yes
    Attrition          
    0          315   90
    1           90  105
    ----------------------------------------------------------------------------------------------------
    교차표 (norm = columns) 
     OverTime         No       Yes
    Attrition                    
    0          0.777778  0.461538
    1          0.222222  0.538462
    ----------------------------------------------------------------------------------------------------
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_38_1.png)
    


    ----------------------------------------------------------------------------------------------------
    카이제곱통계량 58.57149427899665
    p-value 1.9603625783060702e-14
    자유도 1
    기대빈도
     [[273.375 131.625]
     [131.625  63.375]]
    

2) 파악된 내용

- 야근 여부와 관련성이 높다
- 야근이 가장 이직여부에 영향을 많이 끼친다

### 4.숫자→범주


```python
def eda_num_cat(feature, target, data):
    plt.figure(figsize = (12, 4))
    
    plt.subplot(1, 3, 1)
    sns.kdeplot(x= feature, data = data, hue = target,
            common_norm = False)
    
    plt.subplot(1, 3, 2)
    sns.kdeplot(x= feature, data = data, hue = target
            , multiple = 'fill')
    plt.axhline(data[target].mean(), color = 'r')
    
    plt.subplot(1, 3, 3)
    sns.histplot(x= feature, data = data, hue = target
            , multiple = 'fill')
    plt.axhline(data[target].mean(), color = 'r')
    
    plt.tight_layout()
    plt.show()
```

#### (1) Age → Attrition


```python
feature = 'Age'

eda_num_cat(feature, target, data)
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_44_0.png)
    


1) 파악된 내용

- 나이와 이직여부는 관련성이 있어 보인다
- 30세이하에서의 이직율이 높다
    - 계약직 혹은 가치관?
- 30세 이상에서는 이직율이 낮다

#### (2) DistanceFromHome → Attrition


```python
feature = 'DistanceFromHome'

eda_num_cat(feature, target, data)
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_48_0.png)
    


1) 파악된 내용

- 거리도 이직여부와 관련이 조금 있다
- 10 ~ 20, 20 ~ 30 정도의 거리일 때 이직율이 높았다
- 20일 때 이직율이 높지 않았다
    - 자차를 갖고 있는 사람들, 출퇴근 버스 존재 유무 확인 필요

#### (3) MonthlyIncome → Attrition


```python
feature = 'MonthlyIncome'

eda_num_cat(feature, target, data)
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_52_0.png)
    


1) 파악된 내용

- 월급과는 관련이 꽤나 있다
- 2500 달러 이하에서 이직율이 매우 높다
- 7500 ~ 9000 달러 사이는 이직율이 주변 값들에 비해 낮다
    - 직무가 잘 맞을 수도?
* 7000 달러 정도에서 이직율이 높다 -> 이직 사유가 월급 관련된게 아닐수도 있다


#### (4) PercentSalaryHike → Attrition


```python
feature = 'PercentSalaryHike'

eda_num_cat(feature, target, data)
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_56_0.png)
    


1) 파악된 내용

- 관련성이 약간 있다
- 인상률이 22% 이상일 때 이직율이 가장 높았다 (조금 이상) -> 임금 인상률이 이직 사유가 아닐수도 있다
    - 조사 필요

#### (5) TotalWorkingYears → Attrition


```python
feature = 'TotalWorkingYears'

eda_num_cat(feature, target, data)
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis3/output_60_0.png)
    


1) 파악된 내용

- 관련성이 있다
- 오래 일한사람보다 신입일 때 이직율이 높다

### 5.관계 정리하기

① 강한관계

- MaritalStatus, OverTime

② 중간관계

- Age, DFH, MI

③ 관계없음(약한 관계)

- Gender, JS, PSH, TWY
