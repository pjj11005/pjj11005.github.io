---
layout: post
title: KT AIVLE SCHOOL 5기 1주차 | Python
description: KT AIVLE SCHOOL 5기 1주차에 진행한 Python 강의 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## Jupyter notebook

1. 셀 전환 방법
    - 마크 다운으로 전환 : `m`
    - 코드 셀로 전환 : `y`
2. 코드 셀 라인 번호 표시
    - 셀 선택 후 `Shift + L` ⇒ 밑의 모든 셀에도 적용됨
3. 참고 링크
    
[https://gist.github.com/ninanung/2b81a5db946c26c98c573e3662a92b62](https://gist.github.com/ninanung/2b81a5db946c26c98c573e3662a92b62)
    
## 자료형(**List, Dictionary, Tuple**)

1. 리스트(List)
    - 리스트는 여러 값을 하나의 변수로 저장
    - 실전에서는 분석/모델링 할 때 사용하기 보다는, 임시 저장소 같은 역할로 자주 사용

2. 딕셔너리(Dictionary)
    - 딕셔너리는 여러 형태의 자료형을 하나로 묶어 준다 (종합 선물 세트)
    - 실전에서는
        - 데이터 프레임을 만들 때, 딕셔너리를 변환
        - 어떤 함수의 실행 결과가 딕셔너리 형태인 경우, 저장해서 조회용으로 사용
        - JSON 형식이 파이썬에서는 딕셔너리와 호환

3. 튜플(tuple)
    - 튜플은 `좌표, 순서쌍`을 담는 자료형

```python
# 자료형

## 1. 리스트(List)
list(range(1, 12, 2)) # 리스트 안에 리스트 저장 가능

print(a[2:5]) # 2 ~ 4번 인덱스
print(a[2:]) # 2 ~ 끝까지
print(a[-3:]) # 뒤에서 세번째에서 끝까지

del a[2] # 인덱스로 삭제
a.remove(45) # 값으로 삭제

# 리스트로 부터 값 할당
a = [23, 3, 16]
a1, a2, a3 = a

## 2. 딕셔너리(dictionary)

dict_a.items() # key와 값을 쌍(tuple)으로 조회

del dict_a['v2'] # 삭제

# 딕셔너리로 부터 값 할당
dict_a = { 'v1': 32, 'l1': [1,2,3], 'd1': {'a':1, 'b':2}}
# 각 값을 할당
a1, a2, a3 = dict_a.values()


```

## 흐름 제어(조건문과 반복문)

1. Bool 연산자
2. 조건 제어(if - else 문)      
3. 반복 제어
    - `for loop`
        - `range, list, dictionary` 등 으로부터 순서대로 값을 뽑고, 코드를 반복 수행
        - 종료 조건: `range, list, dictionary` 의 값을 모두 추출했을 때, `break`를 만났을 때
    - `while loop`
        - `for loop`와 사용법과 용도는 거의 유사
        - 차이점
            - 조건 변경문이 들어가야 함
            - 종료 조건: 조건문이 `False`일 때, `break`를 만났을 때
        - `무한 반복 & break`
    - 반복 횟수가 정해지지 않을 때는 `while` 사용, 횟수가 정해져 있으면 `for`    

```python
# 흐름 제어

## 1. 조건 제어
math, korean, history=80, 70, 75

if (math < 70) or (korean < 70) or (history < 70): # or: |, and: &
    print('fail')
else:
    print('pass')

## 2. 반복 제어
total = 0
for i in range(1,101): # 모듈로 : 나머지, // : 몫
    if i % 2 == 1:
        total += i
print(total)

multiple7 = [n for n in range(1, 101) if n % 7 == 0] # 리스트 컴프리헨션

dict_a = { 'v1 ' : 32, 'l1' : [1, 2, 3], 'd1' : {'a' : 1, 'b' : 2}}
for key, value in dict_a.items():
    print('key : ', key)
    print('value : ', value)

## 3. 복습 문제: 소수 출력 - 에라토스테네스의 체 이용
#1부터 100까지 수 중, 소수(prime number)를 제외한 모든 수를 출력

### 내 풀이
n = 100
array = [True] * (n+1)

for i in range(2, n + 1):
    if array[i]:
        j = 2
        while i * j <= n:
            array[i * j] = False
            j +=1

for i in range(2, n + 1):
    if not array[i]:
        print(i, end = ',')


### 답안
# 소수가 아닐떄만 바로 출력 (자신보다 작은 수로 나누어 떨어지는 경우)
for i in range(3, 101) :
    for j in range(2, i) :
        if i % j == 0 :
            print(i)
            break
```

- [에라토스테네스의 체](https://velog.io/@changhee09/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%86%8C%EC%88%98%EC%9D%98-%ED%8C%90%EB%B3%84-%EC%97%90%EB%9D%BC%ED%86%A0%EC%8A%A4%ED%85%8C%EB%84%A4%EC%8A%A4%EC%9D%98-%EC%B2%B4) (N보다 작거나 같은 모든 소수 찾을 떄)

    1. 2부터 N까지의 모든 자연수를 나열한다
    2. 남은 수 중에서 아직 처리하지 않은 가장 작은 수 i를 찾는다
    3. 남은 수 중에서 i의 배수를 모두 제거한다(i는 제거하지 않는다)
    4. 더 이상 반복할 수 없을 때까지 2번과 3번 과정을 반복한다.

## 함수 생성 및 활용

### 함수를 만드는 두 가지 목적
- 복잡한 코드 단순화
- 특정 기능 (function)별로 모듈화

### 함수의 구성 요소
1. Input
    - 입력 매개변수
    - 생략 가능
2. 처리
    - 예외 처리
        - `try`: 오류를 감지하고자 하는 코드 블럭
        - `except` 오류 유형: 발생된 오류를 처리
3. Output
    - 실행 결과
    - 생략 가능

```python
#함수 

## 1. Input
def hello(name, loud = 1) :
    if loud == 1 :
        print('HELLO ' + name + ' ~!!!')
    else :
        print('Hello ' + name + ' ~~~')

hello(name = 'Han', loud = 0) # 매개변수 이름과 입력값
hello(name = 'Han')           # 기본값 지정 매개변수 생략
hello('Han')                  # 기본값 지정 매개변수 생략
hello('Han',0)                # 입력매개변수 이름 생략
hello(loud = 0, name = 'Han') # 입력값 순서 바꾸기

def numbers(*nums) : # 입력값의 갯수를 제한하지 않고 받고자 할 때 '*' 사용
    print(nums, type(nums))
    
numbers(1, 2, 3)
numbers(2, 4, 6, 8, 10)# 입력된 데이터 타입은 tuple

## 2. Output
def calculator(a, b) :
    summ = a + b
    mult = a * b
    return summ, mult

r1, r2 = calculator(10, 5)
print(r1, r2)

r1, _ = calculator(10, 5) # 하나의 결과만 필요하다면

b = calculator(1, 3)
print(b, type(b)) # 여러개의 결과를 하나로 받으면 튜플 형태로 받음

## 3. 예외 처리
def divide(a, b) :
    try :
        result = a / b
        return result
    
    except ZeroDivisionError:
        print("0으로 나눌수 없습니다.")
        
divide(10, 0)

## 4. 복습 문제
# 기념일 계산기
from datetime import datetime
def date_check(year, month, day):
    try:
        year = int(year)
        month = int(month)
        day = int(day)

        date = datetime(year, month, day)
        return date
    
    except ValueError as e:
        print("올바른 날짜 형식으로 입력해주세요.")
        return None

def date_count(today):
    birthday = date_check(2021, 12, 2)
    d_day = today - birthday
    return print('오늘은 ', d_day.days, '일째 입니다!')

today = datetime.now()
date_count(today)
```
    
## 데이터 분석/모델링을 위한 데이터 구조

### `CRISP-DM` (데이터 분석 방법론)
- **암기(중요)**
- 거의 모든 데이터 분석 및 ML/DL 관련 프로젝트들의 흐름
- `면접 질문에서 나올정도로 중요함`

![Untitled](/assets/img/blog/KT_AIVLE/week1/Python/006.png)

- Business Understanding(비즈니스 이해): 비즈니스의 문제를 이해하고 정의하는 단계
    - **`비즈니스 관점(도메인 지식)이 매우 중요`**
- Data Understanding (데이터 분석, 이해): 데이터 분석 단계, 가설
- Data Preparation (데이터 전처리)
- Modeling (ML/DL)
- Evaluation (평가): 문제가 해결되었는지 확인
- Deployment (배포)
    
### 분석할 수 있는 데이터
1. 분석할 수 있는 정보의 종류 2가지
    - 숫자 : **연산 가능하면 수치형**
        - 연속형 : 온도 | 이산형 : 판매량, 매출액, 나이
    - 범주 : **묶음 (묶음 안에 공통된 특징 존재), 대체로 우리가 결정해서 만들어냄**
        - 명목형 : 성별, 시도 | 순서형 : 연령대, 매출 등급 
2. 두 가지 동류의 정보가 특별한 구조를 가져야 함
    - 기본 구조: 2차원
        - 행: `분석 단위`, `데이터 한 건, 한 건` → `데이터가 많다 (데이터 행의 수가 많다), 사이즈가 크다 (행과 열 포함 크다)`
            - **행이 어떤 단위인지 정하고 알아내는 과정이 중요함**
        - 열: `정보, 변수, 요인 (x, feature)`, 결과 (`y, target, label`)
3. (분석, 모델링을 위한) 데이터 구조를 다루는 패키지
    - `Numpy`
    - `pandas`

##  Numpy 기초

### 배열 만들기
- 리스트의 장점과 한계, 그리고 `Numpy`
    - 리스트
        - 다른 타입의 데이터 한꺼번에 저장 가능
        - 요소 변경, 추가, 제거 용이
    - 데이터 분석에서는 → `수학적 계산이 가능`하고 `대량의 데이터 처리`가 빨라야 한다
    - `Numpy(Numerical Python)`
        - 빠른 수치 계산을 위해 C언어로 만들어진 Python 라이브러리
        - 벡터와 행렬 연산에 편리한 기능들을 제공
        - 데이터 분석용 라이브러리인 Pandas와 Matplotlib의 기반으로 사용됨
        - Array(벡터, 행렬)단위로 데이터 관리
- 라이브러리 불러오기
    - `Numpy` 라이브러리를 불러와(Import)사용해야 함
- `Numpy` 용어 정리
    - **주요 용어**
        - `axis`: 배열의 각 축
        - `rank`: 축의 개수(차원)
        - `shape`: 축의 길이, 배열의 크기
    - **ex) 3 x 4 배열의 경우**
        - `axis 0` 과 `axis 1` 을 갖는 2차원 배열
        - `rank` 2 array
        - `shape`는 (3, 4)
- 배열 만들기
    - **`np.array()` 함수**를 사용해서 배열을 만든다
    - 대부분 **`리스트`**로부터 배열을 만들거나, 머신러닝 관련 함수 결과 값이 배열이 된다
        
        ![Untitled](/assets/img/blog/KT_AIVLE/week1/Python/007.png)
        
        - 3차원 `shape`: (3, 5) 크기 2차원 배열이 2개이다 라고 많이 읽음
    - **차원 확인**
        - `ndim` 속성으로 배열 차원을 확인
    - **형태(크기) 확인**
        - `shape` 속성으로 배열 형태를 확인
        - 결과는 다음과 같은 형태의 튜플로 표시
            - 1차원: (x, )
            - 2차원: (x, y)
            - 3차원: (x, y, z)
        - 앞에서 부터 `axis 0`, `axis 1`, `axis 2`의 크기를 의미
    - **요소 자료형 확인**
        - `dtype` 속성으로 배열에 포함된 요소들의 자료형을 확인
        - 배열은 한 가지 자료형만 가질 수 있다는 특징이 있다
    - 배열에 대한 이해 - **`Axis 0`의 의미**
        - `데이터`의 건수
        - 2차원 데이터 : (`1000`, 10)
            - 10개의 값(열, 변수)으로 구성된 데이터 `1000건`
            - 분석 대상이 `1000건`, 각 대상은 10개의 정보로 구성됨
        - 3차원 데이터: (`2500`, 28, 28)
            - 28행 28열 (28 X 28) 크기의 2차원 데이터가 `2500건`
            - 만약 흑백 이미지라면 28 X 28 이미지가 `2500장`
    - `Reshape`
        - 배열을 사용할 때 다양한 형태(Shape)로 변환할 필요가 있다
        - 배열에 포함된 **`요소가 사라지지 않는 형태`**라면 자유롭게 변환할 수 있다.
        - (3, 4) → (2, 6) → (4, 3) → (12, 1) → (6, 2) 등등 요소 개수만 변하지 않으면 된다
            - `a.reshape(3, -1)` → a를 3행으로 된 배열로 변환(열의 수는 데이터에 맞게)
            - `a.reshape(-1, 2)` → a를 2열로 된 배열로 변환(행의 수는 데이터에 맞게)
- 함수와 메서드
    - 함수/ 메서드 쓰는 경우
        - ex) 평균 구하기
            - `np.mean()`:
                - 입력 가능한 형태: 리스트, 튜플 등
                - `np array`로 변환해서 평균을 구해 줌
            - 변수(데이터).mean():
                - `np array`만 사용 가능

### 배열 데이터 조회
- 인덱싱
    - 요소 조회
        - `arr1 [행 인덱스, 열 인덱스]`
        - `arr1 [행 인덱스] [열 인덱스]`
    - 행 조회
        - `arr1 [행 인덱스, :]` 혹은 `arr1 [행 인덱스]`
        - `arr1 [[행 1, 행2]]` 혹은 `arr1 [[행 1, 행2], :]`
- 슬라이싱
    - **`배열[행1:행N,열1:열N]`** 형태로 지정해 그 위치의 요소를 조회
    - 조회 결과는 **`2차원 배열`**이 됩니다.
    - 마지막 **범위 값은 대상에 포함되지 않음**
    - 즉, **배열[1:M, 2:N]**이라면 1 ~ M-1행, 2 ~ N-1열이 조회 대상이 됨
- 조건 조회
    - **`조건에 맞는 요소를 선택`**하는 방식이며, **`불리안 방식`**이라고 부름
    - 조회 결과는 **`1차원 배열`**이 됩니다.
    - **`배열[조건]`** 형태로 해당 조건에 맞는 요소만 조회
        - 검색 조건을 변수로 선언해 사용할 수 있다
        - 여러 조건을 **&** 와 **|** 로 연결하여 조회할 수 있다

### 배열 변환과 연산
- 기본 연산 (사칙 연산)
    
    ![Untitled](/assets/img/blog/KT_AIVLE/week1/Python/008.png)
    
- 전치행렬: 주 대각선을 축으로 하는 반사 대칭을 가하여 얻는 행렬
- 배열 집계
    - `np.sum()`, 혹은 `array.sum()`
        - `axis = 0` : 열 기준 집계 (아래 방향)
        - `axis = 1` : 행 기준 집계 (오른쪽 방향)
        - 생략하면 : 전체 집계
    - 동일한 형태로 사용 가능한 함수 : `np.max(), np.min, np.mean(), np.std()`
    
- 자주 사용되는 함수들
    - 가장 큰(작은) 값의 인덱스 반환: `np.argmax`, `np.argmin`
        
        ![Untitled](/assets/img/blog/KT_AIVLE/week1/Python/009.png)
        
    - 조건에 따라 다른 값 지정하기: `np.where` (**중요**)
        - 문법: `np.where(조건문, 참일때값, 거짓일때값)`
            
            ![Untitled](/assets/img/blog/KT_AIVLE/week1/Python/010.png)
                
### 정리
1. 수치 연산을 위해 배열 (array) 를 생성하고 다루는 패키지
2. Array 구조: `Axis, Rank, Shape`
    - 특별히 `Axis 0`의 의미 이해
3. Array 조회
    - 인덱스: 특정 인덱스, 여러 인덱스, 범위 / 조건 조회
4. Array Shape 변형: `reshape`
5. Array 연산
    - 기본 연산: 사칙연산, 지수 , 제곱근
6. Array 집계
    - `sum, mean, min, max, std` … (`axis = 0, 1` 에 따른 연산 방향)
7. 몇가지자주사용하는함수
    - `np.argmax, np.where, …`

##  판다스(Pandas)

### 데이터프레임 생성
- `데이터프레임(Dataframe)`이란?
    - 데이터 분석에서 가장 중요한 데이터 구조
    - 관계형 데이터베이스의 테이블 또는 엑셀 시트와 같은 형태 (`2차원 구조`)
    - 변수들의 집합 → 각 열을 변수라고 부름
    - 열(변수, feature, target) → 의미: `정보`
    - 행(관측치, 샘플) → 의미: `분석단위`
- `시리즈(Series)`이란?
    - 하나의 정보에 대한 데이터들의 집합
    - 데이터프레임에서 `하나의 열을 떼어낸 것`.(`1차원`)
- 딕셔너리로 만들기
    - 딕셔너리로 데이터프레임을 만들면 딕셔너리의 **키**가 **열 이름**이 된다
    - 인덱스를 지정하지 않으면 행 번호가 인덱스가 된다
- CSV파일 읽어오기
    - 분석용 데이터는 대부분 파일에서 읽어 가져오니 잘 익혀야 할 기능이다
    - **`read_csv()`** 함수를 사용해서 CSV 파일에서 데이터를 읽어온다

### 데이터프레임 탐색
- **자주 사용할 만한 메서드들**
    - `head()`: 상위 데이터 확인
    - `tail()`: 하위 데이터 확인
    - `shape`: 데이터프레임 크기
    - `values`: 값 정보 확인(저장하면 2차원 `numpy` 배열이 됨)
    - `columns`: 열 정보 확인
        - 열 이름 리스트 반환
            
            ![Untitled](/assets/img/blog/KT_AIVLE/week1/Python/011.png)
            
    - `dtypes`: 열 자료형 확인
        - int64: 정수형 데이터(int)
        - float64: 실수형 데이터(float)
        - object: 문자열 데이터(string)
    - `info()`: 열에 대한 상세한 정보 확인
    - `describe()`: 기초통계정보 확인
- 정렬해서 보기
    - 인덱스를 기준으로 정렬하는 방법과 특정 열을 기준으로 정렬하는 방법이 있다
    - **`sort_values()`** 메소드로 **특정 열**을 기준으로 정렬
    - **ascending** 옵션을 설정해 오름차순, 내림차순을 설정할 수 있다
        - `ascending=True`: 오름차순 정렬(기본값)
        - `ascending=False`: 내림차순 정렬
    

### 데이터프레임 조회
- 특정 열 조회
    - **`df.loc[ : , [열 이름1, 열 이름2,...]]`** 형태로 조회할 열 이름을 리스트로 지정한다
    - 열 부분은 생략할 수 있었지만, **`행 부분을 생략할 수는 없다`**
    - 하지만 **`df[[열 이름1, 열 이름2,...]]`** 형태로 인덱서를 생략함이 일반적
    - 조회할 열이 하나면 리스트 형태가 아니어도 된다
    
- 조건으로 조회 : **`.loc` (매우 중요)**
    - 단일 조건
        - **`df.loc[조건]`** 형태로 조건을 지정해 조건에 만족하는 데이터만 조회할 수 있다
        - 우선 조건이 제대로 판단이 되는지 확인한 후 그 **조건을 대 괄호 안에** 넣으면 된다
    - 여러 조건 조회
        - [ ]안에 조건을 여러개 연결할 때  `and와 or 대신에 &와 |` 를 사용해야 한다
        - 그리고 각 조건들은 **(조건1) & (조건2)** 형태로 **괄호**로 묶어야 한다
            
            ![Untitled](/assets/img/blog/KT_AIVLE/week1/Python/012.png)
            
    - `isin(), between()` → **아주 유용하다**
        - **`isin([값1, 값2,..., 값n])`**: 값1 또는 값2 또는...값n인 데이터만 조회
            - `주의`: `isin(**리스트**)` 값들을 리스트 형태로 입력해야 함
        - **`between(값1, 값2)`**: 값1 ~ 값2까지 범위안의 데이터만 조회
            - inclusive = 'both' (기본값)
                - 'left', 'right', 'neither' (양쪽의 등호 포함 관련)
    - **조건을 만족하는 행의 일부 열 조회**
        - **`df.loc[조건, ['열 이름1', '열 이름2',...]]`** 형태로 조회할 열을 리스트로 지정 → 2차원, 데이터프레임 형태로 조회
            
### 데이터프레임 집계
- 열 하나 집계
    - 집계하기
        - `dataframe.groupby( ‘집계 기준 변수’, as_index = )[‘집계 대상 변수’].집계 함수`
            - `집계 기준 변수`: ~~별에 해당되는 변수 혹은 리스트. 범주형 변수(예: 월 별, 지역 별등)
            - `집계 대상 변수`: 집계 함수로 집계할 변수 혹은 리스트. (예: 매출액 합계)
        - **`as_index=True`**를 설정(기본값)하면 집계 기준이 되는 열이 인덱스 열이 된다
            - **`as_index=False`**를 설정하면 행 번호를 기반으로 한 정수 값이 인덱스로 설정된다 → False로 두고 사용하자
        - 집계 결과가 data 열만 가지니 **`시리즈`**가 된다
            - `[['data']].sum()`과 같이 하면 열이 여럿이라는 의미여서 결과가 **`데이터프레임`**이 된다
    - 데이터프레임으로 선언
        - 집계 결과를 새로운 데이터프레임으로 선언하여 사용하는 경우가 많다
        - 집계된 결과를 반복해서 사용하거나, 분석 대상이 되는 경우 데이터프레임으로 선언함이 유익하다
        
- 여러 열 집계
    - 여러 열에 대한 집계를 같이 할 수 있다
    - **`[ ['feature1', 'feature2'] ].sum()`** 형태와 같이 집계 대상 열을 리스트로 지정
    - `sum()` 메소드 앞에 아무 열도 지정하지 않으면 **기준열 이외의 모든 열에 대한 집계**가 수행된다
        - 향후에는 이 기능이 제거될 수 있으므로 숫자형 변수만 집계되도록 명시적으로 지정할 필요가 있음
    - **`by=['feature1', 'feature2']`** 과 같이 집계 기준 열을 여럿 설정할 수도 있다

- 여러 함수로 한꺼번에 집계 .agg
    - `df.groupby( ).agg(['함수1','함수2', ...])`

## 출처

- [https://gist.github.com/ninanung/2b81a5db946c26c98c573e3662a92b62](https://gist.github.com/ninanung/2b81a5db946c26c98c573e3662a92b62)
- [https://velog.io/@changhee09/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%86%8C%EC%88%98%EC%9D%98-%ED%8C%90%EB%B3%84-%EC%97%90%EB%9D%BC%ED%86%A0%EC%8A%A4%ED%85%8C%EB%84%A4%EC%8A%A4%EC%9D%98-%EC%B2%B4](https://velog.io/@changhee09/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%86%8C%EC%88%98%EC%9D%98-%ED%8C%90%EB%B3%84-%EC%97%90%EB%9D%BC%ED%86%A0%EC%8A%A4%ED%85%8C%EB%84%A4%EC%8A%A4%EC%9D%98-%EC%B2%B4)