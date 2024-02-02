---
layout: post
title: 2023년 1학기 캡스톤 설계 프로젝트 (CODOG)
description: 2023년 1학기 캡스톤 설계 프로젝트 진행 내용 정리 글입니다.
sitemap: false
---

졸업 작품 프로젝트에서 본인의 진행 작업 위주로 정리해 봤습니다.

* this unordered seed list will be replaced by the toc
{:toc}


## 사이트
코독 - 강아지 친구 매칭 웹사이트(2023.03 ~ 2023.11) ([https://codog.co.kr/](https://codog.co.kr/))

## 프로젝트 개요

> **사용자가 선택한 기준에 따라 이상형 월드컵을 진행하여 원하는 강아지와 매칭 시켜주는 프로젝트입니다.**

### 개발 동기
1. 서비스 부족: 강아지에 특화된 매칭 서비스 부족

2. 애완동물 주인 증가: 반려동물 수요와 관련 서비스 수요 증가

3. 개인정보 안전: 반려동물 주인으로부터 개인 정보를 거의 요구하지 않음

## 프로젝트 사이트맵
![프로젝트 사이트맵](/assets/img/blog/codog/sitemap.png)

## 프로젝트 구현 기능

### 1. 이미지 딥러닝 모델 구현

![이미지 모델](/assets/img/blog/codog/modeling.png)

**1. 데이터**
- 총 4800개의 이미지 (각 품종당 300개)
- 16개 카테고리
- 기타 품종: 퍼그, 아프간 하운드, 사모예드, 셰퍼트, 시베리안 허스키 → 5종류 각각 60개의 이미지로 구성

    ![라벨](/assets/img/blog/codog/label.png)

**2. 데이터 수집:**

- 데이터 구성: 웹 크롤링 데이터 (60%), Tsinghua Dogs 데이터셋 (20%), Kaggle 데이터셋 (20%)을 사용한 데이터 수집
- Tsinghua Dogs 데이터셋: [https://cg.cs.tsinghua.edu.cn/ThuDogs/](https://cg.cs.tsinghua.edu.cn/ThuDogs/)
- Kaggle 데이터셋: [https://www.kaggle.com/competitions/dog-breed-identification/data](https://www.kaggle.com/competitions/dog-breed-identification/data)

**3. 모델 성능 향상:**

   > 기본 베이스 라인 모델 학습 결과가 val_accuracy(약 20%), val_loss가 매우 낮게 나와 모델의 성능을 높이기로 결정

- 데이터 증강: 학습 및 검증 데이터에 회전, 확대/축소, 가로/세로 이동, 가로 반전 등 5-7가지 데이터 증강 기법 적용
- 전이 학습: 13가지 다양한 전이 학습 모델을 사용하여 학습하고, 효율적인 성능을 가진 MobileNetV2 모델을 선택

    ![전이 학습 결과](/assets/img/blog/codog/transferlearning.png)

**4. 모델 성능 평가**
- MobileNetV2 모델 학습 결과 (5 에포크): 모델은 약 92.5%의 val_accuracy와 약 0.3의 val_loss로 수렴

- 전체 모델 정확도: 94.17%

    ![정확도와 손실](/assets/img/blog/codog/accuracy&loss.png)

    ![모델 정확도](/assets/img/blog/codog/modelaccuracy.png)

- 정밀도와 재현율: 대부분의 클래스에 대해 높은 정밀도와 재현율을 보이며, 전반적으로 높은 정확도를 달성. 그러나 비글과 말티즈 클래스의 재현율 값이 비교적 낮음

    ![정밀도와 재현율](/assets/img/blog/codog/precision&recall.png)

- confusion matrix: 히트맵 형식으로 나타낸 결과 같은 클래스 외에는 어둡게 표시

    ![confusionmatrix](/assets/img/blog/codog/confusionmatrix.png)

- 모델 예측: 모델은 4,800개의 다양한 이미지를 매우 잘 예측

    ![모델 예측](/assets/img/blog/codog/modelprediction.png)
    
### 2. 웹사이트 로그인 구현

- 품종 분류: TensorFlow.js로 변환된 학습된 모델을 사용하여 사용자가 품종 예측 여부에 따라 품종 선택 가능
- 패스워드: 중복된 강아지 이름이 발생할 수 있어 패스워드를 중복 방지 처리
- 로그인 구현: 로그인하지 않으면 매칭이 불가능하며, 사용자는 등록 시 제공한 강아지 이름과 패스워드를 사용하여 로그인할 수 있음

    ![품종 분류](/assets/img/blog/codog/breedclassification.png)
    > **품종 분류 결과 화면**


### 3. 매칭 알고리즘

![매칭 알고리즘](/assets/img/blog/codog/matching.png)

- 코사인 유사도 기반으로 알고리즘 수정하여 오류 수정
- firebase DB에 저장된 강아지 정보들을 원활하게 가져오도록 수정
- 토너먼트 진행 후 우승 강아지와 유사한 강아지 3마리 화면에 출력

## 프로젝트 결과

![프로젝트 결과](/assets/img/blog/codog/result.png)

>- 토너먼트 진행 후 우승자 선정
>- 우승자와 유사한 강아지 3마리 추천
>- 강아지 정보를 출력하여 유저와 유저를 연결

## 프로젝트 기대효과

- 강아지를 위한 서비스 부족 해결
    - 강아지에 특화된 매칭 서비스 부족 해결
    - 반려동물 수요와 관련 서비스 수요 증가
- 반려인이 원하는 강아지 선택 가능
    - 백신 및 중성화 된 반려동물을 선택하여 만족도 증가
    - 자신이 원하는 종, 지역을 선택할 수 있어 서비스의 만족도 증가
- 반려인 증가효과 기대
    - 강아지를 위한 서비스 증가로 인한 반려인 증가 기대

## Lesson

- 모델 학습 시 random state를 조절하여 val loss와 accuracy를 효과적으로 수렴시킬 수 있음을 배웠음
- 다양한 모델들을 전이 학습에 활용하면서 각 모델의 특성을 학습하고 활용할 수 있었음
- 데이터 수집 부족으로 모델의 성능 향상이 제한되어 아쉬웠음
- 개와 고양이 두 가지 범주를 대상으로 한다면 서비스의 범용성이 더 넓어졌을 것으로 예상됨

## 팀원

- 지승찬 - 팀장, 서버 관리, 일정 관리 (광운대학교 정보융합학부 데이터사이언스전공)
- 정호빈 - 매칭 알고리즘 구현, 발표 (광운대학교 정보융합학부 데이터사이언스전공)