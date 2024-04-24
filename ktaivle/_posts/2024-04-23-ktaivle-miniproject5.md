---
layout: post
title: 10주차 | 5차 미니프로젝트
description: KT AIVLE SCHOOL 5기 10주차에 진행한 5차 미니프로젝트 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}


> **AICE ASSO 대비 미니 프로젝트**


## 1일차

### AICE 자격 대비 방법

1. AICE 시험에 대한 이해 → 충분하고 정확한 정보
2. 실전과 유사한 연습문제 풀이 → 반복 연습
3. 실전은 연습처럼 편안하게~ → 주어진 시간에 최대로

### 과제 1

```python
df2 = df1.replace('_', np.nan) # 공백 null로 변환

# 중앙값 계산 및 결측치 처리 + 타입 변환
median = df3['age_itg_cd'].median()
df3['age_itg_cd'].fillna(median, inplace=True)
df4 = df3.astype({'age_itg_cd' : 'int64'})
## copy를 먼저 해준 후에 요청사항대로 진행하는 방법도 있음
```

### 과제 2

```python
# 타입 변환
df1['TotalCharges'] = df1['TotalCharges'].astype(float)

# 값 변환(map)
df3['Churn'] = df3['Churn'].map({'No' : 0, 'Yes' : 1})

df4.dropna(inplace=True) # 결측치 행 삭제

# 타입 'object'인 열들 리스트로 저장
object_cols = list(df4.dtypes[df4.dtypes == 'object'].index)
```

## 2일차

### 과제 3

```python
# 특정 열의 특정 값 행 제거
df = df[df['Address1'] != '-']

# 딥러닝 회귀 모델 구조
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(nfeatures, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1)) # activation 함수 없음
```

### 다른 조의 의견

- 정성적, 정량적 평가의 필요성을 강조했다
- 시험 상황에서는 자동 완성, 시선의 처리 등 까다로운 부분이 많다

## 느낀점

- 문제를 직접 코드를 쳐서 구현하는 방식으로 많이 연습하면 시험 합격에 다가갈 수 있을 것 같다