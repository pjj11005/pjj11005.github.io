---
layout: post
title: 재귀(Recursion)
description: 재귀(Recursion) 관련 개념 정리글 입니다.
sitemap: false
---

기본적으로 반복과 순환은 문제 해결 능력이 같으며 많은 경우에 순환 알고리즘을 반복버전으로, 반복 알고리즘을 순환 버전으로 바꾸어 쓸 수 있다.

* this unordered seed list will be replaced by the toc
{:toc}

## 순환 vs 반복

### 순환(Recursion)

- 장점: 순환적인 문제에서는 자연스러운 방법, 기존 반복문에 비해서 코드를 효율적으로 짤 수 있다.

- 단점: 함수 호출의 오버헤드 가능성이 높음, 많은 연산을 필요로 한다. --> 이런한 단점 보완을 위해 동적 프로그래밍을 사용한다.

### 반복(Iteration)

- 장점: for나 while문 이용하여 수행속도가 빠름

- 단점: 순환적인 문제에서는 프로그램 작성이 어려울 수도 있음


## 순환(Recursion) = 재귀 함수

- 정의: 알고리즘이나 함수가 수행 도중에 자기자신을 다시 호출하여 문제를 해결하는 기법

- 특징: 정의 자체가 순환적으로 되어있는 경우에 적합하다.

ex) 팩토리얼, 피보나치 수열, 이항 계수, 이진 탐색, 하노이의 탑,DFS,...


## 재귀 함수 작성 규칙

재귀 함수 작성시 지켜야 하는 아래와 같은 두가지 규칙이 있다.

a. 반복을 멈추는 조건 (Base case) --> 재귀를 멈추는 시점에 대한 조건문

b. 반복할 행위 (Recursion) --> Base case 작성 후 자기 자신을 호출하는 행위


## **참고 문헌 및 사이트** 

- 광운대학교 박재성 교수님의 자료구조 강의 자료

- [https://velog.io/@wkahd01/%EC%9E%AC%EA%B7%80-%ED%95%A8%EC%88%98](https://velog.io/@wkahd01/%EC%9E%AC%EA%B7%80-%ED%95%A8%EC%88%98)

- [https://devraphy.tistory.com/515](https://devraphy.tistory.com/515)