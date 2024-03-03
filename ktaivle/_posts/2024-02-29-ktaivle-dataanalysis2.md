---
layout: post
title: KT AIVLE SCHOOL 5기 2주차 | 데이터 분석(Data analysis)(2)
description: KT AIVLE SCHOOL 5기 2주차에 진행한 데이터 분석(Data analysis) 강의 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}


## 8. 종합실습1 이변량분석(y-숫자) : car seat 매출 분석

* 카시트에 대해서 지역 매장 별 매출액을 예측하고자 합니다.

![carseat](https://cdn.images.express.co.uk/img/dynamic/24/590x/child-car-seat-986556.jpg?r=1532946857754)

### 1.환경준비

#### (1) 라이브러리 로딩


```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as spst
```

#### (2) 데이터 로딩

|	변수명	|	설명	|	구분	|
|----|----|----|
|	Sales 	|	 각 지역 판매량(단위 : 1000개)	|	Target	|
|	CompPrice 	|	지역별 경쟁사 판매가격(달러)	|	feature	|
|	Advertising 	|	 각 지역, 회사의 광고 예산(단위 : 1000달러)	|	feature	|
|	Population 	|	 지역 인구수(단위 : 1000명)	|	feature	|
|	Price 	|	 자사 지역별 판매가격(달러)	|	feature	|
|	ShelveLoc 	|	 진열상태	|	feature	|
|	Age 	|	 지역 인구의 평균 연령	|	feature	|
|	US 	|	 매장이 미국에 있는지 여부	|	feature	|
|	Income 	|	 지역 주민 평균 소득	|	feature	|
|	Urban 	|	 매장이 도시에 있는지 여부	|	feature	|




```python
path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/Carseats2.csv'
data = pd.read_csv(path)
data.head()
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
      <th>Sales</th>
      <th>CompPrice</th>
      <th>Income</th>
      <th>Advertising</th>
      <th>Population</th>
      <th>Price</th>
      <th>ShelveLoc</th>
      <th>Age</th>
      <th>Urban</th>
      <th>US</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.50</td>
      <td>138</td>
      <td>73</td>
      <td>11</td>
      <td>276</td>
      <td>120</td>
      <td>Bad</td>
      <td>42</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.22</td>
      <td>111</td>
      <td>48</td>
      <td>16</td>
      <td>260</td>
      <td>83</td>
      <td>Good</td>
      <td>65</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.06</td>
      <td>113</td>
      <td>35</td>
      <td>10</td>
      <td>269</td>
      <td>80</td>
      <td>Medium</td>
      <td>59</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.40</td>
      <td>117</td>
      <td>100</td>
      <td>4</td>
      <td>466</td>
      <td>97</td>
      <td>Medium</td>
      <td>55</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.15</td>
      <td>141</td>
      <td>64</td>
      <td>3</td>
      <td>340</td>
      <td>128</td>
      <td>Bad</td>
      <td>38</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
target = 'Sales'
```

### 2.숫자형 X → Y

* 모든 숫자형 X에 대해서 Y와 비교하여 차트를 그리고 수치화 하기

① 시각화 : scatter, jointplot  
② 수치화 : 상관분석

함수 생성


```python
def eda_2_nn(var, target, data):
    result = spst.pearsonr(data[var], data[target])
    print(f'상관계수 : {result[0]}, p-value : {result[1]}')
    
    sns.scatterplot(x=var, y = target, data = data)
    plt.grid()
    plt.show()
```

#### (1) Advertising → Sales

1) 시각화 : scatter(regplot), jointplot


```python
var = 'Advertising'

sns.scatterplot(x=var, y = target, data = data)
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_16_0.png)
    



```python
sns.regplot(x=var, y = target, data = data)
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_17_0.png)
    



```python
sns.jointplot(x=var, y = target, data = data)
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_18_0.png)
    


2) 수치화 : 상관분석


```python
result = spst.pearsonr(data[var], data[target])
print(f'상관계수 : {result[0]}, p-value : {result[1]}')
```

    상관계수 : 0.269506781376902, p-value : 4.3776771103027514e-08
    

상관계수 + 산점도 함수


```python
eda_2_nn(var, target, data)
```

    상관계수 : 0.269506781376902, p-value : 4.3776771103027514e-08
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_22_1.png)
    


3) 시각화와 수치화 결과로부터 파악한 내용

* 광고비와 판매량은 관련이 있다. 그러나 크지 않다.

* 추가 : 광고비 0 빼고 분석해보기


```python
temp = data.loc[data['Advertising'] > 0]
sns.scatterplot(x=var, y = target, data = temp)
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_26_0.png)
    



```python
result = spst.pearsonr(temp[var], temp[target])
print(f'상관계수 : {result[0]}, p-value : {result[1]}')
```

    상관계수 : 0.2571829256169392, p-value : 3.112672883086058e-05
    

- 0값을 빼도 딱히 달라지지 않았다...
- 구간을 나눠서 범주형으로 만들면 좋을것 같다

#### (2) Population → Sales

1) 시각화 : scatter(regplot), jointplot

2) 수치화 : 상관분석


```python
var = 'Population'
eda_2_nn(var, target, data)
```

    상관계수 : 0.0504709844720392, p-value : 0.31398160931904856
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_32_1.png)
    


3) 시각화와 수치화 결과로부터 파악한 내용

- 선형적인 관계가 거의 없고 데이터가 고르게 분포 되어있다
- 구간을 나눠서 범주형으로 만들 필요가 있어 보인다
- **카시트를 구매할 연령대의 인구수가 필요하다**

#### (3) Price → Sales

1) 시각화 : scatter(regplot), jointplot

2) 수치화 : 상관분석


```python
var = 'Price'
eda_2_nn(var, target, data)
```

    상관계수 : -0.4449507278465726, p-value : 7.618187011913169e-21
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_38_1.png)
    


3) 시각화와 수치화 결과로부터 파악한 내용

- 자사 가격과 판매량은 관련이 있다 (아주 강한것은 아니다)
- 가격은 100~ 150 달러 사이로 책정하는게 좋아보인다 (데이터 분포가 가장 많다)

#### (4) Age → Sales

1) 시각화 : scatter(regplot), jointplot

2) 수치화 : 상관분석


```python
var = 'Age'
eda_2_nn(var, target, data)
```

    상관계수 : -0.2318154396045765, p-value : 2.7889498393714038e-06
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_44_1.png)
    


3) 시각화와 수치화 결과로부터 파악한 내용

- 지역 평균 나이대와는 큰 관련성이 없다
- 지역 평균 나이도 범주형으로 구간을 나눌 필요가 있어 보인다
- **나이대에 따른 인구구성이 필요하다 (인구수 정보와 마찬가지)**

#### (5) CompPrice → Sales

1) 시각화 : scatter(regplot), jointplot

2) 수치화 : 상관분석


```python
var = 'CompPrice'
eda_2_nn(var, target, data)
```

    상관계수 : 0.06407872955062152, p-value : 0.2009398289418404
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_50_1.png)
    


3) 시각화와 수치화 결과로부터 파악한 내용

- 경쟁사의 가격과 판매량은 큰 관련은 없어보인다
    - 추가 분석 필요
- 하지만 비슷한 가격대인 100 ~140 달러에서도 경쟁사는 많이 팔린 경우가 자사보다 많은 것 같다

4) 추가 분석


```python
data['Price_Diff'] = data['CompPrice'] - data['Price']
data.head()
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
      <th>Sales</th>
      <th>CompPrice</th>
      <th>Income</th>
      <th>Advertising</th>
      <th>Population</th>
      <th>Price</th>
      <th>ShelveLoc</th>
      <th>Age</th>
      <th>Urban</th>
      <th>US</th>
      <th>Price_Diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.50</td>
      <td>138</td>
      <td>73</td>
      <td>11</td>
      <td>276</td>
      <td>120</td>
      <td>Bad</td>
      <td>42</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.22</td>
      <td>111</td>
      <td>48</td>
      <td>16</td>
      <td>260</td>
      <td>83</td>
      <td>Good</td>
      <td>65</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.06</td>
      <td>113</td>
      <td>35</td>
      <td>10</td>
      <td>269</td>
      <td>80</td>
      <td>Medium</td>
      <td>59</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.40</td>
      <td>117</td>
      <td>100</td>
      <td>4</td>
      <td>466</td>
      <td>97</td>
      <td>Medium</td>
      <td>55</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.15</td>
      <td>141</td>
      <td>64</td>
      <td>3</td>
      <td>340</td>
      <td>128</td>
      <td>Bad</td>
      <td>38</td>
      <td>Yes</td>
      <td>No</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
var = 'Price_Diff'
eda_2_nn(var, target, data)
```

    상관계수 : 0.5979217124533921, p-value : 3.877120641788767e-40
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_55_1.png)
    


- 새롭게 만든 경쟁사와 자사의 가격차이 변수가 높은 상관관계를 보인다 (자사 가격보다 높은 상관관계를 보인다)

#### (6) Income → Sales

1) 시각화 : scatter(regplot), jointplot

2) 수치화 : 상관분석


```python
var = 'Income'
eda_2_nn(var, target, data)
```

    상관계수 : 0.15195097946754424, p-value : 0.0023096704539517904
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_60_1.png)
    


3) 시각화와 수치화 결과로부터 파악한 내용

- 평균 소득과도 큰 관련성이 없어 보인다
- 구간을 나눴을 때 약간의 관련성이 보인다 → 범주형으로 나누면 좋을것 같다

### 3.범주형 X → Y

* 모든 범주형 X에 대해서 Y와 비교하여 차트를 그리고 수치화 하기

① 시각화 : 평균비교 barplot  
② 수치화 : t-test, anova

#### (1) ShelveLoc → Sales

1) 시각화 : 평균비교 barplot


```python
var = 'ShelveLoc'

sns.barplot(x=var, y=target, data = data)
plt.grid()
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_67_0.png)
    


2) 수치화 : t-test, anova


```python
d_b = data.loc[data[var] == 'Bad', target]
d_m = data.loc[data[var] == 'Medium', target]
d_g = data.loc[data[var] == 'Good', target]

spst.f_oneway(d_b, d_m, d_g)
```




    F_onewayResult(statistic=92.22990509910348, pvalue=1.26693609015938e-33)



3) 시각화와 수치화 결과로부터 파악한 내용

- 진열상태에 따라 판매량에 큰 차이가 있다
- 진열상태를 좋은 상태로 유지할 필요가 있다

#### (2) Urban → Sales

1) 시각화 : 평균비교 barplot


```python
var = 'Urban'

sns.barplot(x=var, y=target, data = data)
plt.grid()
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_74_0.png)
    


2) 수치화 : t-test, anova


```python
urban = data.loc[data[var] == 'Yes', target]
not_urban = data.loc[data[var] == 'No', target]

spst.ttest_ind(not_urban, urban)
```




    TtestResult(statistic=0.30765346670661126, pvalue=0.7585069603942775, df=398.0)



3) 시각화와 수치화 결과로부터 파악한 내용

유의미한 차이가 없다 → 도시여부와 판매량은 관련이 적다

#### (3) US → Sales

1) 시각화 : 평균비교 barplot


```python
var = 'US'

sns.barplot(x = var, y = target, data = data)
plt.grid()
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_analysis2/output_81_0.png)
    


2) 수치화 : t-test, anova


```python
not_us= data.loc[data[var] == 'Yes', target]
us = data.loc[data[var] == 'No', target]

spst.ttest_ind(not_us, us)
```




    TtestResult(statistic=3.589738747802499, pvalue=0.00037233958701471517, df=398.0)



3) 시각화와 수치화 결과로부터 파악한 내용

약간의 차이가 있다 → 미국인지 아닌지는 판매량과 약간의 관련이 있다

### 4.관계 정리하기

① 강한관계

- Price_Diff, ShelveLoc

② 중간관계

- Price, Income, Age, US, Advertising

③ 관계없음

- Population, CompPrice, Urban

- 관련성이 적다고 무작정 삭제하는 것이 아니다
    - 나중에 유의미한 변수 생성 가능하기 때문 
