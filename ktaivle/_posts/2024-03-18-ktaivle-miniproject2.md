---
layout: post
title: KT AIVLE SCHOOL 5기 5주차 | 2차 미니프로젝트
description: KT AIVLE SCHOOL 5기 5주차에 진행한 2차 미니프로젝트 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}


## 1일차

### 주제

- 공공 데이터 활용한 미세먼지 농도 예측
    - 서울 지역의 미세먼지 데이터와 날씨 데이터를 활용하여, 미세먼지 예측에 관련 있는 데이터 항목으로 데이터를 구성, 전처리 하여 미세먼지 농도를 예측하는 머신러닝 모델 구현

### 데이터 전처리 및 분석

- 시계열 데이터로 `time` 변수를 기준으로 미세먼지, 날씨 데이터 `merge`
- 1시간 후 미세먼지 농도를 `target` 으로 설정
- 시계열 데이터로 `ffill, bfill` 사용 → 놓친 부분
- 날씨 데이터는 생각보다 무의미 했고, 미세 먼지 및 대기 오염 물질 정보가 관련이 깊었다

### 모델링

- `Linear Regression, Random Forest, Gradient Boost, XGBoost, LightGBM` 모델 사용
- `MSE, R2`평가 결과 `Linear Regression, Random Forest` 모델의 성능이 가장 좋았다
- 변수 중요도는 `PM10` 변수가 압도적이고 `전날 미세먼지, CO, PM25, NO2`가 다음으로 높았다