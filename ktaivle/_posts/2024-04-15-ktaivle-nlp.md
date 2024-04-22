---
layout: post
title: 9주차 | 언어지능 딥러닝
description: KT AIVLE SCHOOL 5기 9주차에 진행한 언어지능 딥러닝 강의 내용 정리 글입니다.
sitemap: false
---

* this unordered seed list will be replaced by the toc
{:toc}

## TF-IDF, 비지도학습, Word Embedding

- 자연어 vs 형식 언어
    - 자연어 : 사람의 언어 → 유연하다
    - 형식 언어 : 기호를 넣어 사용하는 언어(ex : 프로그래밍 언어)

### TF-IDF(Term Frequency - Inverse Document Frequency)

> 주어진 문서를 요약할 때 어떠한 부분을 가지고 요약을 하는 지에 대한 논의
> 
> 
> $$TF(t, d) * IDF(t, D)$$
> 

- 단어 표현 (Word Representation, Word Embedding, Word Vector)
    - 자연어 처리 모델에 적용하기 위한 방법
        - 언어적인 특성을 반영하여 단어를 수치화 하는 방법 → 벡터
    - 데이터 표현
        - 기본 : One-Hot Encoding → **자연어 단어 표현에는 부적합**
            - 단어의 수가 많으므로 고차원 저밀도 벡터를 구성하게 됨
            - 새로운 단어 표현이 힘들다, 연관성 표현이 힘들다
        - 분포 가설에 기반하여 **벡터의 크기가 작으면서 단어의 의미를 표현하는 법**
            - 분포 가설 : 비슷한 위치에 나오는 단어 = 비슷한 의미
            - 분포 가설 기반의 두 가지 데이터 표현법
                - 카운트 기반 방법(Count-based): 특정 문맥 안에서 단어들이 동시에 등장하는 횟수를 직접 셈
                - 예측 방법(Predictive): 신경망 등을 통해 문맥 안의 단어들을 예측
    - 단어 표현
        - 정의 : 텍스트가 얼마나 유사한지를 표현하는 방식
        - 유사도 판단 방식
            - 같은 단어의 개수를 사용해서 유사도를 판단하는 방법
            - 형태소로 나누어 형태소를 비교하는 방법
        - 딥러닝 기반 유사도 판단
            - 텍스트 벡터화 후, 벡터화된 각 문장 간의 유사도를 측정하는 방식
            - 대표적인 유사도 측정 방식 : 자카드 유사도, 코사인 유사도, 유클리디언 유사도, 맨하탄 유사도
            - 유사도 측정 전에 단어 벡터화 필요(TF-IDF 활용)
                - 자카드 유사도 제외한 모든 유사도 판단에서 사용
                - 자카드 유사도는 벡터화 없이 바로 유사도 측정 가능
                
                ```python
                ## similarity_cosine
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                sentence = ('오늘도 폭염이 이어졌는데요, 내일은 반가운 비 소식이 있습니다.',
                            '폭염을 피해 놀러왔다가 갑작스런 비로 망연자실하고 있습니다.')
                vector = TfidfVectorizer(max_features=100) # 사이즈가 100인 벡터 생성
                tfidf_vector = vector.fit_transform(sentence)
                print(cosine_similarity(tfidf_vector[0], tfidf_vector[0]))
                print(cosine_similarity(tfidf_vector[0], tfidf_vector[1]))
                
                ## similarity_jaccard : 순서 유무에 따른 버전이 두가지 있음
                import numpy as np
                from sklearn.metrics import accuracy_score
                
                a = np.array([1, 3, 2])
                b = np.array([1, 4, 5])
                c = np.array([4, 1, 5])
                d = np.array([1, 1, 0, 0])
                e = np.array([1, 1, 0, 2])
                f = np.array([1, 0, 1, 0])
                
                print(accuracy_score(a, b)) # 전체 요소중에서 같은 원소의 확률
                print(accuracy_score(b, c))
                print(accuracy_score(a, c))
                print(accuracy_score(d, e))
                print(accuracy_score(d, f))
                ```
                

### 비지도학습

- Clustering
    - Introduction
        - Classification vs. Clustering
            - Classification : 미리 라벨링 된 데이터로 클래스 예측(지도 학습)
            - Clustering : 라벨링 되지 않은 데이터로 그룹으로 구별하는 것(비지도 학습)
        - 추천 시스템, 검색 엔진, 자연어 처리…
    - Data Types and Representations
        - Discrete vs. Continuous
            - Discrete : 하나의 값만 가진 것
            - Continuius : 연속적인 값(변하는 값)
        - Data representations
            - Data matrix (object-by-feature structure) → 데이터 그 자체
            - Distance/dissimilarity matrix (object-by-object structure) → 데이터 간의 거리
    - Distance Measures
        - Minkowski Distance
            - 두 벡터 간의 거리(p=1: 맨하탄, p=2 : 유클리디안)
            - 분야에 따라서 p값을 다르게 사용
        - Cosine Measure
            - d = 1 - cos(x, y)
            - 0 ≤ d ≤ 2 → 관련이 적을수록 거리가 멀다
    - Major Clustering Approaches
        - Partitioning Approach
            - Typical methods: K-means(단점 : k의 수를 미리 정해줘야 한다), K-medoids, CLARANS,…
            - KNN (k Nearest Neighbor) → 전체 개수를 모르기 때문에 k값에 영향을 받게 된다
        - Clustering Ensemble Approach
            - 다양한 clustering 결과를 합침 (다른 k값 분류들)
            - Typical methods: Evidence-accumulation based, graph-based…

### Word Embedding

> 단어들을 재구성 해주는 작업
> 
> 
> 분포 가설을 이용하여 단어를 조합 시켜서 만듦
> 

- Word2Vec
    - one hot encoding : text를 벡터로 변환
        - 유사도 판단이 불가능하다
        - 고차원 저밀도라서 학습하기 힘들다
    - Embedding
        - **dense vector with similarity**
    - **Word2Vec**
        - word embedding, Similarity comes from neighbor words
        - windows size에 따라 neighbor words 달라짐(windows size는 관련이 있을 거리를 의미)
            - windows size=2, 해당 단어와 좌우로 2칸 내에 인접한 단어들이 neighbor words가 된다
        - embedding을 통해 생성된 벡터값을 사용

## 정보 검색, 추천 시스템

### 추천 시스템의 배경과 목적

- 파레토 법칙 vs 롱테일 법칙
    - 파레토의 법칙: 상위 20%가 80%의 가치를 창출한다
    - 롱테일의 법칙: 하위 80%의 다수가 상위 20%보다 뛰어난 가치를 창출한다
- 추천 시스템의 정의
    - 사용자의 행동 이력, 사용자 간 관계, 상품 유사도, 사용자 컨텍스트에 기반하여 사용자의 관심 상품을 자동으로 예측하고 제공하는 시스템
- 추천 알고리즘 개요
    - **Collaborative Filtering(CF)** : 컨텐츠를 벡터로 만들어, 유사도로 판단(코사인 유사도)
        - Cosine Similarity 방식(내적 기반의 유사도 연산 필수)
        - 단점 : 아이템의 존재 유무에 따라서 메트릭이 다 바뀐다
    - **Content-based Filtering(CBF)** : 내용 자체의 키워드가 중요함
        - TF-IDF 방식(원하는 키워드가 존재하는 지에 대한 카운팅이 필수)
    - 최근에는 두 가지 방식을 결합하여 사용

## 언어지능 신경망

### Introduction

- 딥러닝 학습 및 추론 과정
    1. Deep Learning Model setup
    2. Training (with Large-scale Dataset)
    3. Inference / Testing (Real-Word Execution)
- Overfitting 방지 방법
    - 더 많은 training data
    - Autoencoding(or variational auto-encoder (VAE))
    - Dropout
    - Regularization
- 딥러닝 모델의 종류
    - CNN, RNN,…

### Linear Functions

1. Linear Regression
    - training vs learning
        - training: 데이터를 통해 최적의 파라미터를 찾는 과정
        - learning: 새로운 데이터에 대해서 일반화를 잘하게 하는 과정
2. Binary Classification
3. Softmax Classification

### Nonlinear Functions

1. ANN
    - 인공 신경망의 단점
        1. 레이어를 많이 쌓지 못했다 → ReLU
        2. 2차원 이상 처리 불가하다 → CNN

### Advanced Topics

1. GAN(Generative Adversarial Network)
    - generator, discriminator 두 가지 학습 → 두 가지를 경쟁하여 성능 향상
    - Discriminator Model: 순수 이진 분류기(진위 판별) → 지도 학습
    - Generator Model: train data의 분포를 학습 → 비지도 학습
        - 학습 데이터의 분포를 따라가도록 학습 수행
        - Noise → Generator Model → Image
    - G의 목적 함수 → max G
    - D의 목적 함수 → min G, max D

1. PCA / LDA
    - 차원 축소: 목적에 맞게 W값을 잘 조절하여 차원을 축소하자
    - PCA: 최대한 원래의 분포를 유지하며 차원 축소
    - LDA: 차원 축소 시켰을 때 분류가 잘되게 하는 방법

1. Overfitting
    - 방지 방법
        1. 더 많은 데이터로 학습
        2. Reduce the number of features → **Autoencoding**, Dropout, Regularization
            - Autoencoding: 최대한 핵심 정보를 포함 시키도록 차원 축소하여 학습
            - 일반적으로 Dropout 사용

## RNN

- Sequence Data
    - 단어를 이전 단어 + 다음 단어(time series)에 기반하여 이해한다
- 이전 값과 input을 이용하여 다음 값을 예측해 나간다
- RNN applications
    - Language Modeling, Speech Recognition, Machine Translation, Question Answering (QA) Systems, Conversation Modeling, Image/Video Captioning, Image/Music/Dance Generation
    - one to many: Image Captioning
    - many to one: Sentiment Classification
    - many to many: Machine Translation, Video Classification on frame level

## 강화 학습

- RL: 학습 수행 반복과 보상을 통해 학습하는 과정
- DRL: 신경망을 활용하여 가장 성능이 좋은 함수 선택

![Untitled](/assets/img/blog/KT_AIVLE/week9/001.png)

## LLM

- GPT(Generative Pre-Trained)
- 세대 별 발전
    - 1세대 : 딥러닝 이전 기술/태스크에 딥러닝 적용
        - 미리 정의한 label중 top-1출력 방식, 응용 태스크 마다 별도의 학습 데이터 구축
    - 2세대: 대용량 데이터로부터 일반적 지식을 (사전)학습
        1. 응용 태스크 마다 별도의 학습 데이터 필요
        2. 사전 학습 모델의 품질이 응용 태스크의 성능 좌우
    - 3세대: 초거대 AI 모델의 가능성 탐색
        - 사전 학습 모델을 추가 학습 없이 그래로 활용(필요 시, 예제 제공)
- 트랜스포머
    - 인코더, 디코더로 이루어짐
    - 학습을 위한 인코딩, 추론을 위한 디코딩
- 강화 학습 기반 LLM
    1. 신경망을 통한 학습 (Pre-trained)
    2. 강화 학습을 통해 성능 개선 및 추론
    - reward를 기반으로 학습하고 맞춰나가는 경향이 있음
- GPT3
    - (글쓰기 능력) 뉴스 기사 생성
    - (산술 추론) Arithmetic
    - (언어 추론) Learning and Using Novel Words
- InstructGPT
    - (1단계) Pre-trained Model → (2단계) Instruction-tuned Model
    - 2단계 : 사용자 의도 일치(alignment) 학습
        - rule-base 기반
- 향후 연구 방향
    - multi-agent, personalization, robot intelligence

## 정리

- TF를 보고 겹치는게 많으면 유사
    - Contents-Based Filtering
    - 자카드 유사도
- 벡터로 만들어, 각도가 좁으면 유사(코사인 유사도)
    - Collaborative Filtering
    - 코사인 유사도