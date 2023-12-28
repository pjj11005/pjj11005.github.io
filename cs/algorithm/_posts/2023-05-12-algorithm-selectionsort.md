---
layout: post
title: 선택 정렬(Selection Sort)
categories: 
  - cs
  - algorithm
description: 선택 정렬(Selection Sort) 관련 개념 정리글 입니다.
sitemap: false
---

선택 정렬은 정렬 알고리즘 중 가장 기본적이면서 간단한 알고리즘 중 하나이다. 

* this unordered seed list will be replaced by the toc
{:toc}

##  정의 
선택 정렬은 정렬되지 않은 리스트에서 가장 작은 값을 찾아 첫 번째 위치와 교환하고, 그 다음으로 작은 값을 찾아 두 번째 위치와 교환하는 과정을 반복하여 정렬하는 알고리즘이다. 

##  동작 원리

1. 우선 주어진 리스트 중 최소값을 찾는다.
2. 그 값을 맨 앞에 위치한 값과 교체한다.
3. 맨 처음 위치를 뺀 나머지 리스트를 같은 방법으로 교체하여 반복해 나간다.

![Selection Sort](/assets/img/blog/selectionsort.png){: width="800" height="400"}

Selection Sort 동작 방식
{:.figure}

##  특징

- 장점

    - 간단하고 이해하기 쉽다.
    - 안정 정렬 알고리즘이다.
    - 작은 데이터 세트와 잘 작동한한다.
    - 다양한 유형의 데이터 유형에 적응할 수 있다.
    - 선택 정렬은 내부 정렬 알고리즘이므로 목록을 정렬하는 데 추가 메모리가 필요하지 않는다.
    - 하드웨어로 쉽게 구현할 수 있어 실시간 응용 프로그램에 적합하다.

- 단점

    - 선택 정렬은 최악의 경우와 평균적인 경우에 $$O(n^{2})$$의 시간 복잡도를 가진다.
    - 큰 데이터 세트에서는 제대로 작동하지 않는다.

##  구현 코드

~~~python
# Python program for implementation of Selection

import sys
A = [64, 25, 12, 22, 11]

# Traverse through all array elements
for i in range(len(A)):
	
	# Find the minimum element in remaining
	# unsorted array
	min_idx = i
	for j in range(i+1, len(A)):
		if A[min_idx] > A[j]:
			min_idx = j
			
	# Swap the found minimum element with
	# the first element	
	A[i], A[min_idx] = A[min_idx], A[i]

# Driver code to test above
print ("Sorted array")
for i in range(len(A)):
	print("%d" %A[i],end=" , ")

#output
# Sorted array: 
# 11 12 22 25 64 
~~~

선택 정렬 구현 코드 예시 (시간 복잡도 $$O(n^{2})$$)
{:.figure}


## 참고 문헌 및 사이트

- [https://www.geeksforgeeks.org/selection-sort/](https://www.geeksforgeeks.org/selection-sort/)

- [https://hongcoding.tistory.com/181](https://hongcoding.tistory.com/181)

- chat gpt