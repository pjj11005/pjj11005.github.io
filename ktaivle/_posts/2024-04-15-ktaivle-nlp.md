---
layout: post
title: 9주차 | 언어지능 딥러닝(1)
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
> $TF(t, d) * IDF(t, D)$
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