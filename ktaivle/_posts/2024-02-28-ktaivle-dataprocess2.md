---
layout: post
title: KT AIVLE SCHOOL 5기 2주차 | 데이터 처리(Data processing)(2)
description: KT AIVLE SCHOOL 5기 2주차에 진행한 데이터 처리(Data processing) 강의 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## 8. 종합실습 단변량 분석

### 1. 비즈니스 시나리오

![car](https://cdn.images.express.co.uk/img/dynamic/24/590x/child-car-seat-986556.jpg?r=1532946857754)

* **고객사는 카시트를 판매하는 회사**
* 최근 매출 하락에 대해 각 부서가 파악한 원인
    * 최근에 경쟁사와의 가격 경쟁력이 하락하고 있고, 광고비용이 적절하게 집행되지 않음
    * 너무 국내 시장에 집중됨
    * 지역마다의 구매력을 제대로 파악하지 못하고 있음
>
* 데이터에 근거하여 문제 해결
    * 단변량 분석을 통해 비즈니스 인사이트를 도출하기

### 2. 환경준비

#### (1) 라이브러리 불러오기


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
```

#### (2) 데이터 로딩 : 카시트 판매량 데이터

|	변수명	|	설명	|	구분	|
|----|----|----|
|	**Sales** 	|	 각 지역 판매량(단위 : 1000개)	|	Target	|
|	**CompPrice** 	|	 경쟁사 가격(단위 : 달러)	|	feature	|
|	**Income** 	|	 지역 평균 소득(단위 : 1000달러)	|	feature	|
|	**Advertising** 	|	 각 지역, 회사의 광고 예산(단위 : 1000달러)	|	feature	|
|	**Population** 	|	 지역 인구수(단위 : 1000명)	|	feature	|
|	**Price** 	|	 자사 지역별 판매가격	|	feature	|
|	**ShelveLoc** 	|	 진열상태	|	feature	|
|	**Age** 	|	 지역 인구의 평균 연령	|	feature	|
|	**Urban** 	|	 도심 지역 여부(Yes,No)	|	feature	|
|	**US** 	|	 매장이 미국에 있는지 여부(Yes,No)	|	feature	|



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



### 3.단변량분석 : 숫자형 변수

* 숫자형변수를 분석하는 함수를 생성


```python
def eda_1_num(data, var, bins = 30):

    # 기초통계량
    print('<< 기초통계량 >>')
    display(data[[var]].describe().T)
    print('=' * 100)

    # 시각화
    print('<< 그래프 >>')
    plt.figure(figsize = (10,6))

    plt.subplot(2,1,1)
    sns.histplot(data[var], bins = bins, kde = True)
    plt.grid()

    plt.subplot(2,1,2)
    sns.boxplot(x = data[var])
    plt.grid()
    plt.show()
```

#### (1) Sales (Target)


```python
var = 'Sales'
```

1) 변수의 비즈니스 의미

* 각 지역 판매량(단위 : 1000개)
* Target

2) 기초통계량 및 분포확인


```python
eda_1_num(data, var)
```

    << 기초통계량 >>
    


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
      <th>Sales</th>
      <td>400.0</td>
      <td>7.496325</td>
      <td>2.824115</td>
      <td>0.0</td>
      <td>5.39</td>
      <td>7.49</td>
      <td>9.32</td>
      <td>16.27</td>
    </tr>
  </tbody>
</table>
</div>


    ====================================================================================================
    << 그래프 >>
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_18_3.png)
    


3) 기초통계량과 분포를 통해서 파악한 내용

>보이는 그대로를 넘어, 비즈니스 관점에서 적기

* 대부분의 판매량이 5천개 ~ 만개 사이
* 만개 이상의 판매량이 다수 존재한다

4) 추가 분석사항

* 판매량 구간을 0 ~ 4천, 4천 ~ 1만, 1만 이상 으로 구분해서 분석해 볼 필요 있음

#### (2) CompPrice


```python
var = 'CompPrice'
```

1) 변수의 비즈니스 의미

- 경쟁사 가격 (단위: 달러)

2) 기초통계량 및 분포확인


```python
eda_1_num(data, var)
```

    << 기초통계량 >>
    


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
      <th>CompPrice</th>
      <td>400.0</td>
      <td>124.975</td>
      <td>15.334512</td>
      <td>77.0</td>
      <td>115.0</td>
      <td>125.0</td>
      <td>135.0</td>
      <td>175.0</td>
    </tr>
  </tbody>
</table>
</div>


    ====================================================================================================
    << 그래프 >>
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_28_3.png)
    


3) 기초통계량과 분포를 통해서 파악한 내용

>보이는 그대로를 넘어, 비즈니스 관점에서 적기

- 가격이 110 ~ 140 달러 사이에 많이 분포해있다
- 80달러 이하, 160달러 이상의 가격이 있는 이유 -> 가격 정책에 대해서 알아봐야 한다
    - ex) 팔리지 않는 제품을 가격을 낮춰서 판매 등...
- 제품 종류에 가격대가 차이가 나지 않을까?

4) 추가 분석해 볼 사항

- 구간을 나누어 분석해볼 필요가 있다
- 자사 가격과 비교 :
    - 전체 분포 비교
    - 지역별 가격 비교

#### (3) Income


```python
var = 'Income'
```

1) 변수의 비즈니스 의미

- 지역 평균 소득(단위 : 1000달러)
- feature

2) 기초통계량 및 분포확인


```python
eda_1_num(data, var)
```

    << 기초통계량 >>
    


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
      <th>Income</th>
      <td>400.0</td>
      <td>68.6575</td>
      <td>27.986037</td>
      <td>21.0</td>
      <td>42.75</td>
      <td>69.0</td>
      <td>91.0</td>
      <td>120.0</td>
    </tr>
  </tbody>
</table>
</div>


    ====================================================================================================
    << 그래프 >>
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_38_3.png)
    


3) 기초통계량과 분포를 통해서 파악한 내용

>보이는 그대로를 넘어, 비즈니스 관점에서 적기

- 대체로 분포가 고르다 -> 카시트는 모든 소득 수준에서 필요한 필수품의 성격이다
- 2만 ~ 4만, 6만 ~ 10만의 수입을 가진 지역들이 많다 (멀티 모달 그래프 -> 봉우리가 2개)
* 전 연령에 대한 인구별 평균 소득액? 성인에 대한 평균 소득액?
* 2만달러 ~ 12만 달러 이므로 연간 소득일 것.
* 50%가 약 4만달러 ~ 9만달러

4) 추가 분석해 볼 사항

* 평균 소득과 카시트 판매량과의 관계
* 평균 소득과 카시트 가격과의 관계

#### (4) Advertising


```python
var = 'Advertising'
```

1) 변수의 비즈니스 의미

- 각 지역, 회사의 광고 예산(단위 : 1000달러)

2) 기초통계량 및 분포확인


```python
eda_1_num(data, var)
```

    << 기초통계량 >>
    


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
      <th>Advertising</th>
      <td>400.0</td>
      <td>6.635</td>
      <td>6.650364</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>12.0</td>
      <td>29.0</td>
    </tr>
  </tbody>
</table>
</div>


    ====================================================================================================
    << 그래프 >>
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_48_3.png)
    


3) 기초통계량과 분포를 통해서 파악한 내용

>보이는 그대로를 넘어, 비즈니스 관점에서 적기

- 대부분의 경우 광고비를 사용하지 않았으므로 사용한 부분들 확인할 필요가 있다
- 예산 관련 정보를 알아볼 필요가 있다
* 광고비 예산이 0 인 지역이 전체에 36%나 됨

4) 추가 분석해 볼 사항

- 광고비 0이 아닌 데이터들을 분석해볼 필요가 있다


```python
temp = data.loc[data['Advertising'] > 0]
```


```python
eda_1_num(temp, 'Advertising', bins = 20)
```

    << 기초통계량 >>
    


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
      <th>Advertising</th>
      <td>256.0</td>
      <td>10.367188</td>
      <td>5.509879</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>29.0</td>
    </tr>
  </tbody>
</table>
</div>


    ====================================================================================================
    << 그래프 >>
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_54_3.png)
    


#### (5) Population


```python
var = 'Population'
```

1) 변수의 비즈니스 의미

- 지역 인구수(단위 : 1000명)

2) 기초통계량 및 분포확인


```python
eda_1_num(data, var)
```

    << 기초통계량 >>
    


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
      <th>Population</th>
      <td>400.0</td>
      <td>264.84</td>
      <td>147.376436</td>
      <td>10.0</td>
      <td>139.0</td>
      <td>272.0</td>
      <td>398.5</td>
      <td>509.0</td>
    </tr>
  </tbody>
</table>
</div>


    ====================================================================================================
    << 그래프 >>
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_60_3.png)
    


3) 기초통계량과 분포를 통해서 파악한 내용

>보이는 그대로를 넘어, 비즈니스 관점에서 적기

- 14 ~ 40만명에 50% 분포
- 거의 균등하게 분포

4) 추가 분석해 볼 사항

- 인구수에 따른 판매량, 평균 연령의 관계

#### (6) Price


```python
var = 'Price'
```

1) 변수의 비즈니스 의미

- 자사 지역별 판매가격 (달러)

2) 기초통계량 및 분포확인


```python
eda_1_num(data, var)
```

    << 기초통계량 >>
    


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
      <th>Price</th>
      <td>400.0</td>
      <td>115.795</td>
      <td>23.676664</td>
      <td>24.0</td>
      <td>100.0</td>
      <td>117.0</td>
      <td>131.0</td>
      <td>191.0</td>
    </tr>
  </tbody>
</table>
</div>


    ====================================================================================================
    << 그래프 >>
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_70_3.png)
    


3) 기초통계량과 분포를 통해서 파악한 내용

>보이는 그대로를 넘어, 비즈니스 관점에서 적기

- 100 ~ 135 달러 사이에 많이 분포
- 경쟁 회사에 비해서 범위를 벗어난 가격들이 많다 -> 가격 책정을 제대로 못하는 것 같다
- 경쟁사에 비해서 저렴하게 판매하고 있다
- 가격대가 24 ~ 191 달러로 경쟁사보다 범위가 넓다

4) 추가 분석해 볼 사항

- 가격에 따라 판매량은 달라질까?
- 경쟁사와의 가격 비교가 필요하다


```python
# 경쟁사와 자사 가격의 비교
plt.figure(figsize = (10,6))
sns.kdeplot(data['Price'], label = 'Price')
sns.kdeplot(data['CompPrice'], label = 'CompPrice')

price_mean = round(np.mean(data['Price']),1)
com_mean = round(np.mean(data['CompPrice']),1)

plt.axvline(price_mean, color = 'C0', linewidth = .5)
plt.axvline(com_mean, color = 'C1', linewidth = .5)

plt.text(price_mean-16, 0.001, price_mean, color = 'C0')
plt.text(com_mean+5, 0.001, com_mean, color = 'C1')

plt.legend()
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_75_0.png)
    



```python
plt.figure(figsize = (10,6))
sns.boxplot(data = data[['Price', 'CompPrice']], orient = 'h')
plt.show()
```


    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_76_0.png)
    


#### (7) Age


```python
var = 'Age'
```

1) 변수의 비즈니스 의미

- 지역 인구의 평균 연령

2) 기초통계량 및 분포확인


```python
eda_1_num(data, var)
```

    << 기초통계량 >>
    


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
      <th>Age</th>
      <td>400.0</td>
      <td>53.3225</td>
      <td>16.200297</td>
      <td>25.0</td>
      <td>39.75</td>
      <td>54.5</td>
      <td>66.0</td>
      <td>80.0</td>
    </tr>
  </tbody>
</table>
</div>


    ====================================================================================================
    << 그래프 >>
    


    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_82_3.png)
    


3) 기초통계량과 분포를 통해서 파악한 내용

>보이는 그대로를 넘어, 비즈니스 관점에서 적기

- 40 ~ 65세의 나이대가 많다
- 대체로 분포가 고르다

* 지역별 평균나이는 거의 균등분포로 보임.
* 균등분포 ==> 임의로 이렇게 모았거나, 여러 분포가 섞였거나, 이상치를 처리했거나...

* 평균나이
    * 지역 인구 전체에 대한 평균
    * 카시트가 필요한 사람은, 약 10세 이하 자녀를 둔 부모
    * 10세 이하 자녀들과 20대후반 ~ 40세의 부모가 주축인 인구의 평균 나이는?

4) 추가 분석해 볼 사항

* 가장 판매량이 높은 평균나이는?

### 4.단변량분석 : 범주형 변수

* 함수 생성


```python
def eda_1_cat(data, var) :
    # 기초통계량
    t1 = data[var].value_counts()
    t2 = data[var].value_counts(normalize = True)
    t3 = pd.concat([t1, t2], axis = 1)
    t3.columns = ['count','ratio']
    display(t3)
    
    # 그래프
    sns.countplot(x = var, data = data)
    plt.show()
    
```

#### (1) ShelveLoc


```python
var = 'ShelveLoc'
```

1) 변수의 비즈니스 의미

* 매장 진열상태

2) 기초통계량(수치화)


```python
eda_1_cat(data, var)
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
      <th>ratio</th>
    </tr>
    <tr>
      <th>ShelveLoc</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Medium</th>
      <td>219</td>
      <td>0.5475</td>
    </tr>
    <tr>
      <th>Bad</th>
      <td>96</td>
      <td>0.2400</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>85</td>
      <td>0.2125</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_95_1.png)
    


3) 기초통계량과 분포를 통해서 파악한 내용

>보이는 그대로를 넘어, 비즈니스 관점에서 적기

* 진열 상태가 중간 정도 가 54%

4) 추가 분석해 볼 사항

* 진열상태에 따른 판매량 차이
* 지역, 인구수와 진열 상태 관계 비교

#### (2) Urban


```python
var = 'Urban'
```

1) 변수의 비즈니스 의미

- 도심 지역 여부(Yes,No)

2) 기초통계량(수치화)


```python
eda_1_cat(data, var)
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
      <th>ratio</th>
    </tr>
    <tr>
      <th>Urban</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Yes</th>
      <td>282</td>
      <td>0.705</td>
    </tr>
    <tr>
      <th>No</th>
      <td>118</td>
      <td>0.295</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_105_1.png)
    


3) 기초통계량과 분포를 통해서 파악한 내용

>보이는 그대로를 넘어, 비즈니스 관점에서 적기

- 도심 지역이 70% 이상
- 도심 지역에 아이를 키우는 가정이 많을 확률이 높아서 구매력, 판매량이 높을 가능성이 있음

4) 추가 분석해 볼 사항

- 도심지와 그렇지 않은 곳에서의 판매량 차이
- 도심지 여부에 따른 지역 소득, 인구수와의 관련성 비교
* 도시와 지방의 인구수, 소득수준, 자사 가격, 경쟁사 가격, 광고예산, 판매량 비교 필요

#### (3) US


```python
var = 'US'
```

1) 변수의 비즈니스 의미

- 매장이 미국에 있는지 여부(Yes,No)

2) 기초통계량(수치화)


```python
eda_1_cat(data, var)
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
      <th>ratio</th>
    </tr>
    <tr>
      <th>US</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Yes</th>
      <td>258</td>
      <td>0.645</td>
    </tr>
    <tr>
      <th>No</th>
      <td>142</td>
      <td>0.355</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](/assets/img/blog/KT_AIVLE/week2/data_process/output_115_1.png)
    


3) 기초통계량과 분포를 통해서 파악한 내용

>보이는 그대로를 넘어, 비즈니스 관점에서 적기

- 매장이 미국인 곳이 64% 이상
- 해외에서는 어떤 옵션의 제품들이 잘 판매되는지 파악 필요

4) 추가 분석해 볼 사항이 있나요?

- 도심 지역 여부, 매장 미국 여부, 판매량, 가격의 관련성 파악 필요
- 인구 수와 지역 소득과의 관련성도 파악 필요
* 국내외 와 광고비, 판매가, 경쟁사가격, 매출액 등 비교
